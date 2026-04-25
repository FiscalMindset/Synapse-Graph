from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .inference import (
    AttentionTrace,
    GenerationRequest,
    HeadMask,
    InferenceResponse,
    ModelTopology,
    NeuralInferenceEngine,
)
from .om_client import NeuralCatalogBinding, OpenMetadataNeuralMapper, OpenMetadataSettings

LOGGER = logging.getLogger(__name__)


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYNAPSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_title: str = "Synapse-Graph Neural Proxy"
    api_description: str = (
        "A FastAPI neural proxy that bridges local generation, mechanistic tracing, and "
        "OpenMetadata lineage for Synapse-Graph."
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class OpenMetadataStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    connected: bool
    catalog_ready: bool
    defective_heads: list[HeadMask] = Field(default_factory=list)
    last_defect_sync_at: datetime | None = None
    last_ingest_error: str | None = None


class SessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: datetime
    prompt: str
    response_text: str
    trace: AttentionTrace
    masked_heads: list[HeadMask] = Field(default_factory=list)


class StateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topology: ModelTopology | None = None
    latest_session: SessionSnapshot | None = None
    masked_heads: list[HeadMask] = Field(default_factory=list)
    ollama_available: bool = False
    openmetadata: OpenMetadataStatus


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    response: InferenceResponse
    masked_heads: list[HeadMask] = Field(default_factory=list)
    topology: ModelTopology
    openmetadata: OpenMetadataStatus


class LocalHeadMaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int = Field(ge=0)
    head_index: int = Field(ge=0)


class NeuralProxyRuntime:
    def __init__(self) -> None:
        self.service_settings = ServiceSettings()
        self.openmetadata_settings = OpenMetadataSettings()
        self.inference = NeuralInferenceEngine()
        self.openmetadata = OpenMetadataNeuralMapper(self.openmetadata_settings)

        self._catalog_binding: NeuralCatalogBinding | None = None
        self._topology: ModelTopology | None = None
        self._topology_lock = asyncio.Lock()
        self._lineage_tasks: set[asyncio.Task[Any]] = set()

        self._latest_session: SessionSnapshot | None = None
        self._last_defect_sync_at: datetime | None = None
        self._last_ingest_error: str | None = None
        self._openmetadata_connected: bool = False
        self._preload_task: asyncio.Task[Any] | None = None

    async def startup(self) -> None:
        await self.inference.startup()
        LOGGER.info(
            "Resolved models: ollama=%s, hf=%s, use_ollama=%s",
            self.inference.settings.ollama_model,
            self.inference.settings.hf_model_name,
            self.inference.settings.use_ollama_if_available,
        )
        if self.openmetadata_settings.openmetadata_enabled:
            self._openmetadata_connected = await self.openmetadata.is_available()
        # Kick off a background attempt to preload the HuggingFace tracer so the UI
        # can display real topology without blocking startup. This will attempt the
        # configured HF model and the configured fallback inside the tracer.
        if not self._preload_task:
            self._preload_task = asyncio.create_task(self._background_preload())

    async def _background_preload(self) -> None:
        try:
            await self.inference._hooked_runner.ensure_loaded()
            try:
                topo = await self.inference.get_model_topology()
                async with self._topology_lock:
                    self._topology = topo
            except Exception:
                LOGGER.exception("Failed to refresh topology after HF preload.")

            # If OpenMetadata is enabled, attempt to bootstrap the catalog now that
            # we may have a real topology.
            if self.openmetadata_settings.openmetadata_enabled and self._catalog_binding is None:
                try:
                    async with self._topology_lock:
                        if self._topology is not None:
                            self._catalog_binding = await self.openmetadata.ensure_catalog(self._topology)
                            self._openmetadata_connected = True
                            self._last_ingest_error = None
                except Exception as exc:
                    self._openmetadata_connected = False
                    self._last_ingest_error = str(exc)
                    self._catalog_binding = None
                    LOGGER.exception("OpenMetadata bootstrap failed during background preload.")
        except Exception:
            LOGGER.exception("Background HF preload failed (non-fatal).")

    async def shutdown(self) -> None:
        for task in list(self._lineage_tasks):
            task.cancel()
        for task in list(self._lineage_tasks):
            with suppress(asyncio.CancelledError):
                await task
        if self._preload_task is not None:
            self._preload_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._preload_task
        await self.openmetadata.close()
        await self.inference.shutdown()

    async def get_state(self) -> StateResponse:
        topology, catalog = await self.ensure_topology()
        if (
            self.openmetadata_settings.openmetadata_enabled
            and catalog is not None
            and not self.inference.get_masked_heads()
        ):
            await self.sync_defective_heads(catalog)

        return StateResponse(
            topology=topology,
            latest_session=self._latest_session,
            masked_heads=self.inference.get_masked_heads(),
            ollama_available=await self.inference.is_ollama_available(),
            openmetadata=self._openmetadata_status(catalog_ready=catalog is not None),
        )

    async def ensure_topology(self) -> tuple[ModelTopology, NeuralCatalogBinding | None]:
        if self._topology is not None and (
            not self.openmetadata_settings.openmetadata_enabled or self._catalog_binding is not None
        ):
            return self._topology, self._catalog_binding

        async with self._topology_lock:
            if self._topology is None:
                # Avoid eager HF model load unless explicitly requested via preload_shadow_model.
                try:
                    if self.inference.settings.preload_shadow_model:
                        self._topology = await self.inference.get_model_topology()
                    else:
                        ollama_available = await self.inference.is_ollama_available()
                        model_name = (
                            self.inference.settings.ollama_model
                            if ollama_available and self.inference.settings.use_ollama_if_available
                            else self.inference.settings.hf_model_name
                        )
                        self._topology = ModelTopology(
                            model_name=model_name,
                            device=("ollama" if ollama_available else "cpu"),
                            total_layers=0,
                            total_heads=0,
                            layers=[],
                        )
                except Exception:
                    LOGGER.exception("Failed to determine model topology; falling back to placeholder.")
                    self._topology = ModelTopology(
                        model_name=self.inference.settings.hf_model_name,
                        device="unknown",
                        total_layers=0,
                        total_heads=0,
                        layers=[],
                    )

            if self.openmetadata_settings.openmetadata_enabled and self._catalog_binding is None:
                try:
                    self._catalog_binding = await self.openmetadata.ensure_catalog(self._topology)
                    self._openmetadata_connected = True
                    self._last_ingest_error = None
                except Exception as exc:
                    LOGGER.exception("OpenMetadata catalog bootstrap failed.")
                    self._last_ingest_error = str(exc)
                    self._openmetadata_connected = False
                    self._catalog_binding = None

        return self._topology, self._catalog_binding

    async def sync_defective_heads(
        self,
        catalog: NeuralCatalogBinding | None,
    ) -> list[HeadMask]:
        if catalog is None:
            return self.inference.get_masked_heads()

        try:
            defective_set = await self.openmetadata.sync_defective_heads(catalog)
            masked_heads = await self.inference.set_masked_heads(defective_set)
            self._last_defect_sync_at = _utcnow()
            self._openmetadata_connected = True
            self._last_ingest_error = None
            return masked_heads
        except Exception as exc:
            LOGGER.exception("Failed to synchronize defective heads from OpenMetadata.")
            self._last_ingest_error = str(exc)
            self._openmetadata_connected = False
            return self.inference.get_masked_heads()

    async def set_local_head_mask(self, request: LocalHeadMaskRequest) -> list[HeadMask]:
        current = {
            (mask.layer_index, mask.head_index)
            for mask in self.inference.get_masked_heads()
        }
        current.add((request.layer_index, request.head_index))
        return await self.inference.set_masked_heads(current)

    async def clear_local_head_masks(self) -> list[HeadMask]:
        return await self.inference.set_masked_heads(set())

    async def _select_trace_model(self, request: GenerationRequest) -> None:
        if not request.trace_model_name:
            return
        if request.trace_model_name == self.inference.settings.hf_model_name:
            return

        async with self._topology_lock:
            await self.inference.set_analysis_model(request.trace_model_name)
            self._topology = None
            self._catalog_binding = None
            self._last_ingest_error = None

    async def generate(self, request: GenerationRequest) -> GenerateResponse:
        await self._select_trace_model(request)
        topology, catalog = await self.ensure_topology()
        masked_heads = await self.sync_defective_heads(catalog)
        session_id = str(uuid4())

        async def step_listener(step: Any) -> None:
            if catalog is None:
                return
            self._schedule_lineage_ingest(catalog, session_id, request.prompt, step)

        response = await self.inference.generate(request, step_listener=step_listener)
        session = SessionSnapshot(
            session_id=session_id,
            created_at=_utcnow(),
            prompt=request.prompt,
            response_text=response.text,
            trace=response.trace,
            masked_heads=masked_heads,
        )
        self._latest_session = session

        return GenerateResponse(
            session_id=session_id,
            response=response,
            masked_heads=masked_heads,
            topology=topology,
            openmetadata=self._openmetadata_status(catalog_ready=catalog is not None),
        )

    async def stream_response(self, request: GenerationRequest) -> StreamingResponse:
        await self._select_trace_model(request)
        topology, catalog = await self.ensure_topology()
        masked_heads = await self.sync_defective_heads(catalog)
        session_id = str(uuid4())
        event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        async def step_listener(step: Any) -> None:
            await event_queue.put(
                (
                    "trace_step",
                    {
                        "sessionId": session_id,
                        "step": step.model_dump(mode="json"),
                    },
                )
            )
            if catalog is not None:
                self._schedule_lineage_ingest(catalog, session_id, request.prompt, step)

        async def producer() -> None:
            output_chunks: list[str] = []
            try:
                async for chunk in self.inference.stream(request, step_listener=step_listener):
                    if chunk.token:
                        output_chunks.append(chunk.token)
                        await event_queue.put(
                            (
                                "token",
                                {
                                    "sessionId": session_id,
                                    "token": chunk.token,
                                },
                            )
                        )

                    if chunk.done and chunk.trace is not None:
                        session = SessionSnapshot(
                            session_id=session_id,
                            created_at=_utcnow(),
                            prompt=request.prompt,
                            response_text="".join(output_chunks),
                            trace=chunk.trace,
                            masked_heads=masked_heads,
                        )
                        self._latest_session = session
                        await event_queue.put(
                            (
                                "done",
                                {
                                    "sessionId": session_id,
                                    "responseText": session.response_text,
                                    "trace": chunk.trace.model_dump(mode="json"),
                                    "maskedHeads": [mask.model_dump(mode="json") for mask in masked_heads],
                                },
                            )
                        )
                        return
            except Exception as exc:
                LOGGER.exception("Streaming generation failed.")
                await event_queue.put(
                    (
                        "error",
                        {
                            "sessionId": session_id,
                            "message": str(exc),
                        },
                    )
                )
            finally:
                await event_queue.put(("close", {}))

        async def event_iterator() -> Any:
            producer_task = asyncio.create_task(producer())
            try:
                yield _format_sse(
                    "session",
                    {
                        "sessionId": session_id,
                        "topology": topology.model_dump(mode="json"),
                        "maskedHeads": [mask.model_dump(mode="json") for mask in masked_heads],
                        "openmetadata": self._openmetadata_status(
                            catalog_ready=catalog is not None
                        ).model_dump(mode="json"),
                    },
                )

                while True:
                    event_name, payload = await event_queue.get()
                    if event_name == "close":
                        break
                    yield _format_sse(event_name, payload)
            finally:
                with suppress(asyncio.CancelledError):
                    await producer_task

        return StreamingResponse(event_iterator(), media_type="text/event-stream")

    def _schedule_lineage_ingest(
        self,
        catalog: NeuralCatalogBinding,
        session_id: str,
        prompt: str,
        step: Any,
    ) -> None:
        task = asyncio.create_task(self._ingest_lineage_task(catalog, session_id, prompt, step))
        self._lineage_tasks.add(task)
        task.add_done_callback(self._lineage_tasks.discard)

    async def _ingest_lineage_task(
        self,
        catalog: NeuralCatalogBinding,
        session_id: str,
        prompt: str,
        step: Any,
    ) -> None:
        try:
            await self.openmetadata.ingest_step(catalog, session_id, prompt, step)
            self._openmetadata_connected = True
            self._last_ingest_error = None
        except Exception as exc:
            self._last_ingest_error = str(exc)
            self._openmetadata_connected = False
            LOGGER.exception("OpenMetadata lineage ingestion failed.")

    def _openmetadata_status(self, *, catalog_ready: bool) -> OpenMetadataStatus:
        return OpenMetadataStatus(
            enabled=self.openmetadata_settings.openmetadata_enabled,
            connected=self._openmetadata_connected,
            catalog_ready=catalog_ready,
            defective_heads=self.inference.get_masked_heads(),
            last_defect_sync_at=self._last_defect_sync_at,
            last_ingest_error=self._last_ingest_error,
        )


runtime = NeuralProxyRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    await runtime.startup()
    yield
    await runtime.shutdown()


app = FastAPI(
    title=runtime.service_settings.api_title,
    description=runtime.service_settings.api_description,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=runtime.service_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=StateResponse)
async def healthcheck() -> StateResponse:
    return await runtime.get_state()


@app.get("/api/v1/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    return await runtime.get_state()


@app.post("/api/v1/openmetadata/bootstrap", response_model=StateResponse)
async def bootstrap_openmetadata() -> StateResponse:
    topology, catalog = await runtime.ensure_topology()
    if catalog is not None:
        await runtime.sync_defective_heads(catalog)
    return StateResponse(
        topology=topology,
        latest_session=runtime._latest_session,
        masked_heads=runtime.inference.get_masked_heads(),
        ollama_available=await runtime.inference.is_ollama_available(),
        openmetadata=runtime._openmetadata_status(catalog_ready=catalog is not None),
    )


@app.post("/api/v1/openmetadata/sync-defects", response_model=StateResponse)
async def sync_defects() -> StateResponse:
    topology, catalog = await runtime.ensure_topology()
    if catalog is not None:
        await runtime.sync_defective_heads(catalog)
    return StateResponse(
        topology=topology,
        latest_session=runtime._latest_session,
        masked_heads=runtime.inference.get_masked_heads(),
        ollama_available=await runtime.inference.is_ollama_available(),
        openmetadata=runtime._openmetadata_status(catalog_ready=catalog is not None),
    )


@app.post("/api/v1/governance/local-mask", response_model=StateResponse)
async def set_local_head_mask(request: LocalHeadMaskRequest) -> StateResponse:
    topology, catalog = await runtime.ensure_topology()
    await runtime.set_local_head_mask(request)
    return StateResponse(
        topology=topology,
        latest_session=runtime._latest_session,
        masked_heads=runtime.inference.get_masked_heads(),
        ollama_available=await runtime.inference.is_ollama_available(),
        openmetadata=runtime._openmetadata_status(catalog_ready=catalog is not None),
    )


@app.post("/api/v1/governance/clear-local-masks", response_model=StateResponse)
async def clear_local_head_masks() -> StateResponse:
    topology, catalog = await runtime.ensure_topology()
    await runtime.clear_local_head_masks()
    return StateResponse(
        topology=topology,
        latest_session=runtime._latest_session,
        masked_heads=runtime.inference.get_masked_heads(),
        ollama_available=await runtime.inference.is_ollama_available(),
        openmetadata=runtime._openmetadata_status(catalog_ready=catalog is not None),
    )


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerationRequest) -> GenerateResponse:
    return await runtime.generate(request)


@app.post("/api/v1/generate/stream")
async def generate_stream(request: GenerationRequest) -> StreamingResponse:
    return await runtime.stream_response(request)


@app.post("/api/v1/hf/preload", response_model=StateResponse)
async def preload_hf() -> StateResponse:
    """Force-load the HuggingFace shadow tracer and refresh the cached topology.

    Useful when the configured HF model failed to load at startup and we want to
    attempt a fallback or to pre-warm the tracer for visualization.
    """
    try:
        # Ensure the hooked runner is loaded (may attempt fallback inside _load_sync)
        await runtime.inference._hooked_runner.ensure_loaded()
        # Refresh topology cache
        async with runtime._topology_lock:
            try:
                runtime._topology = await runtime.inference.get_model_topology()
            except Exception:
                runtime._topology = None
    except Exception:
        LOGGER.exception("HF preload failed.")
    return await runtime.get_state()


def _format_sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
