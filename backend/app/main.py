from __future__ import annotations

import asyncio
import difflib
from itertools import combinations
import json
import time
import logging
import re
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import quote

from .inference import (
    AttentionTrace,
    GenerationRequest,
    HeadMask,
    InferenceResponse,
    ModelTopology,
    NeuralInferenceEngine,
    TraceExecutionMode,
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


class CausalAutopsyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    trace_model_name: str = "gpt2"
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    layer_index: int | None = Field(default=None, ge=0)
    head_index: int | None = Field(default=None, ge=0)


class CausalAutopsyTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    layer_name: str
    head_index: int
    head_name: str
    selection_reason: str


class CausalAutopsyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: CausalAutopsyTarget
    baseline: InferenceResponse
    ablated: InferenceResponse
    text_similarity: float
    causal_effect_score: float
    verdict: str
    interpretation: str


class CircuitDiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    target_hallucination_token: str = Field(min_length=1)
    system_prompt: str | None = None
    trace_model_name: str = "gpt2"
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    top_k_heads: int = Field(default=5, ge=1, le=20)
    max_pair_sweeps: int = Field(default=10, ge=0, le=190)


class CircuitHead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    layer_name: str
    head_index: int
    head_name: str
    activation_score: float


class CircuitAblationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    masked_heads: list[CircuitHead]
    output_text: str
    target_present: bool
    target_count: int
    text_similarity: float
    causal_effect_score: float
    trace: AttentionTrace | None = None


class CircuitDiscoveryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_hallucination_token: str
    baseline: InferenceResponse
    baseline_target_present: bool
    baseline_target_count: int
    candidate_heads: list[CircuitHead]
    sweep_results: list[CircuitAblationResult]
    discovered_circuit: list[CircuitHead]
    combined_causal_effect: float
    verdict: str


class QuarantineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heads: list[CircuitHead]
    reason: str | None = None


class OpenMetadataWebhookResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: bool
    parsed_heads: list[HeadMask] = Field(default_factory=list)
    masked_heads: list[HeadMask] = Field(default_factory=list)
    reason: str


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

    async def run_causal_autopsy(self, request: CausalAutopsyRequest) -> CausalAutopsyResponse:
        await self._select_trace_model(
            GenerationRequest(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=0,
                top_p=0.95,
                stream=False,
                execution_mode=TraceExecutionMode.FAITHFUL,
                trace_model_name=request.trace_model_name,
            )
        )
        await self.ensure_topology()

        previous_masks = {
            (mask.layer_index, mask.head_index)
            for mask in self.inference.get_masked_heads()
        }

        baseline_request = GenerationRequest(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=0,
            top_p=0.95,
            stream=False,
            execution_mode=TraceExecutionMode.FAITHFUL,
            trace_model_name=request.trace_model_name,
        )

        try:
            await self.inference.set_masked_heads(set())
            baseline = await self.inference.generate(baseline_request)
            target = _select_autopsy_target(
                baseline.trace,
                layer_index=request.layer_index,
                head_index=request.head_index,
            )

            await self.inference.set_masked_heads({(target.layer_index, target.head_index)})
            ablated = await self.inference.generate(baseline_request)
        finally:
            await self.inference.set_masked_heads(previous_masks)

        text_similarity = difflib.SequenceMatcher(None, baseline.text, ablated.text).ratio()
        causal_effect_score = max(0.0, min(1.0, 1.0 - text_similarity))
        verdict = _causal_verdict(causal_effect_score)

        return CausalAutopsyResponse(
            target=target,
            baseline=baseline,
            ablated=ablated,
            text_similarity=round(text_similarity, 4),
            causal_effect_score=round(causal_effect_score, 4),
            verdict=verdict,
            interpretation=(
                f"Masking {target.layer_name}:{target.head_name} changed the deterministic replay "
                f"with effect score {causal_effect_score:.3f}. {verdict}"
            ),
        )

    async def discover_circuit(self, request: CircuitDiscoveryRequest) -> CircuitDiscoveryResponse:
        start_time = time.monotonic()
        LOGGER.info(
            "discover_circuit: start (top_k=%s, max_pair_sweeps=%s)",
            request.top_k_heads,
            request.max_pair_sweeps,
        )

        generation_request = GenerationRequest(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=0,
            top_p=0.95,
            stream=False,
            execution_mode=TraceExecutionMode.FAITHFUL,
            trace_model_name=request.trace_model_name,
        )
        await self._select_trace_model(generation_request)
        await self.ensure_topology()

        previous_masks = {
            (mask.layer_index, mask.head_index)
            for mask in self.inference.get_masked_heads()
        }

        try:
            await self.inference.set_masked_heads(set())
            t0 = time.monotonic()
            LOGGER.info("discover_circuit: baseline generation starting")
            baseline = await self.inference.generate(generation_request)
            t1 = time.monotonic()
            LOGGER.info("discover_circuit: baseline generation completed (%.3fs)", t1 - t0)
            candidate_heads = _rank_circuit_heads(baseline.trace, top_k=request.top_k_heads)
            baseline_target_count = _count_target_token(
                baseline.text,
                request.target_hallucination_token,
            )

            mask_groups: list[tuple[CircuitHead, ...]] = [(head,) for head in candidate_heads]
            pair_groups = list(combinations(candidate_heads, 2))[: request.max_pair_sweeps]
            mask_groups.extend(pair_groups)

            sweep_results: list[CircuitAblationResult] = []
            LOGGER.info("discover_circuit: running %d sweep groups", len(mask_groups))
            for idx, mask_group in enumerate(mask_groups, start=1):
                LOGGER.info(
                    "discover_circuit: sweep %d/%d - masks=%s",
                    idx,
                    len(mask_groups),
                    [(h.layer_index, h.head_index) for h in mask_group],
                )
                step_t0 = time.monotonic()
                await self.inference.set_masked_heads(
                    {
                        (head.layer_index, head.head_index)
                        for head in mask_group
                    }
                )
                ablated = await self.inference.generate(generation_request)
                step_t1 = time.monotonic()
                LOGGER.info(
                    "discover_circuit: sweep %d generation completed (%.3fs)",
                    idx,
                    step_t1 - step_t0,
                )
                target_count = _count_target_token(
                    ablated.text,
                    request.target_hallucination_token,
                )
                text_similarity = difflib.SequenceMatcher(None, baseline.text, ablated.text).ratio()
                target_reduction = _target_reduction_score(
                    baseline_target_count=baseline_target_count,
                    ablated_target_count=target_count,
                )
                causal_effect_score = max(
                    0.0,
                    min(1.0, max(1.0 - text_similarity, target_reduction)),
                )
                sweep_results.append(
                    CircuitAblationResult(
                        masked_heads=list(mask_group),
                        output_text=ablated.text,
                        target_present=target_count > 0,
                        target_count=target_count,
                        text_similarity=round(text_similarity, 4),
                        causal_effect_score=round(causal_effect_score, 4),
                        trace=ablated.trace,
                    )
                )
            LOGGER.info("discover_circuit: completed sweeps; total so far (%.3fs)", time.monotonic() - t0)
        finally:
            await self.inference.set_masked_heads(previous_masks)

        best_result = _select_best_circuit_result(
            sweep_results,
            baseline_target_count=baseline_target_count,
        )
        discovered_circuit = best_result.masked_heads if best_result is not None else []
        combined_effect = best_result.causal_effect_score if best_result is not None else 0.0

        return CircuitDiscoveryResponse(
            target_hallucination_token=request.target_hallucination_token,
            baseline=baseline,
            baseline_target_present=baseline_target_count > 0,
            baseline_target_count=baseline_target_count,
            candidate_heads=candidate_heads,
            sweep_results=sweep_results,
            discovered_circuit=discovered_circuit,
            combined_causal_effect=combined_effect,
            verdict=_circuit_verdict(
                baseline_target_count=baseline_target_count,
                best_result=best_result,
                target_hallucination_token=request.target_hallucination_token,
            ),
        )

    async def discover_circuit_stream(self, request: CircuitDiscoveryRequest) -> StreamingResponse:
        """Run circuit discovery and stream progress via Server-Sent Events.

        Events emitted:
        - session: initial topology and metadata
        - progress: generic progress messages
        - sweep_start: {idx, total, masks}
        - sweep_done: {idx, result_summary}
        - done: final CircuitDiscoveryResponse as JSON string
        - error: on failure
        """
        generation_request = GenerationRequest(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=0,
            top_p=0.95,
            stream=False,
            execution_mode=TraceExecutionMode.FAITHFUL,
            trace_model_name=request.trace_model_name,
        )

        await self._select_trace_model(generation_request)
        topology, catalog = await self.ensure_topology()

        event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        async def producer() -> None:
            previous_masks = {
                (mask.layer_index, mask.head_index)
                for mask in self.inference.get_masked_heads()
            }
            try:
                await self.inference.set_masked_heads(set())
                await event_queue.put(("progress", {"message": "baseline_generation_start"}))
                t0 = time.monotonic()
                baseline = await self.inference.generate(generation_request)
                t1 = time.monotonic()
                await event_queue.put(("progress", {"message": "baseline_generation_done", "duration": t1 - t0}))

                candidate_heads = _rank_circuit_heads(baseline.trace, top_k=request.top_k_heads)
                baseline_target_count = _count_target_token(baseline.text, request.target_hallucination_token)

                mask_groups: list[tuple[CircuitHead, ...]] = [(head,) for head in candidate_heads]
                pair_groups = list(combinations(candidate_heads, 2))[: request.max_pair_sweeps]
                mask_groups.extend(pair_groups)

                sweep_results: list[CircuitAblationResult] = []
                await event_queue.put(("progress", {"message": "running_sweeps", "count": len(mask_groups)}))

                for idx, mask_group in enumerate(mask_groups, start=1):
                    await event_queue.put(("sweep_start", {"idx": idx, "total": len(mask_groups), "masks": [(h.layer_index, h.head_index) for h in mask_group]}))
                    step_t0 = time.monotonic()
                    await self.inference.set_masked_heads({(h.layer_index, h.head_index) for h in mask_group})
                    ablated = await self.inference.generate(generation_request)
                    step_t1 = time.monotonic()
                    await event_queue.put(("sweep_done", {
                        "idx": idx,
                        "duration": step_t1 - step_t0,
                        "target_count": _count_target_token(ablated.text, request.target_hallucination_token),
                    }))

                    target_count = _count_target_token(ablated.text, request.target_hallucination_token)
                    text_similarity = difflib.SequenceMatcher(None, baseline.text, ablated.text).ratio()
                    target_reduction = _target_reduction_score(
                        baseline_target_count=baseline_target_count,
                        ablated_target_count=target_count,
                    )
                    causal_effect_score = max(
                        0.0,
                        min(1.0, max(1.0 - text_similarity, target_reduction)),
                    )
                    sweep_results.append(
                        CircuitAblationResult(
                            masked_heads=list(mask_group),
                            output_text=ablated.text,
                            target_present=target_count > 0,
                            target_count=target_count,
                            text_similarity=round(text_similarity, 4),
                            causal_effect_score=round(causal_effect_score, 4),
                            trace=ablated.trace,
                        )
                    )

                best_result = _select_best_circuit_result(sweep_results, baseline_target_count=baseline_target_count)
                discovered_circuit = best_result.masked_heads if best_result is not None else []
                combined_effect = best_result.causal_effect_score if best_result is not None else 0.0

                final = CircuitDiscoveryResponse(
                    target_hallucination_token=request.target_hallucination_token,
                    baseline=baseline,
                    baseline_target_present=baseline_target_count > 0,
                    baseline_target_count=baseline_target_count,
                    candidate_heads=candidate_heads,
                    sweep_results=sweep_results,
                    discovered_circuit=discovered_circuit,
                    combined_causal_effect=combined_effect,
                    verdict=_circuit_verdict(
                        baseline_target_count=baseline_target_count,
                        best_result=best_result,
                        target_hallucination_token=request.target_hallucination_token,
                    ),
                )

                # Send final response as JSON string for client to parse
                await event_queue.put(("done", {"response": final.model_dump(mode="json")}))
            except Exception as exc:
                LOGGER.exception("discover_circuit_stream producer failed")
                await event_queue.put(("error", {"message": str(exc)}))
            finally:
                await self.inference.set_masked_heads(previous_masks)
                await event_queue.put(("close", {}))

        async def event_iterator() -> Any:
            producer_task = asyncio.create_task(producer())
            try:
                yield _format_sse(
                    "session",
                    {
                        "topology": topology.model_dump(mode="json") if topology is not None else None,
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

    async def quarantine_circuit(self, request: QuarantineRequest) -> OpenMetadataWebhookResponse:
        topology, catalog = await self.ensure_topology()
        if not self.openmetadata_settings.openmetadata_enabled or catalog is None:
            return OpenMetadataWebhookResponse(
                applied=False,
                parsed_heads=[],
                masked_heads=self.inference.get_masked_heads(),
                reason="OpenMetadata is not enabled or the catalog is not available.",
            )

        # Build pair set from provided heads
        parsed_pairs: set[tuple[int, int]] = {
            (h.layer_index, h.head_index) for h in request.heads
        }

        try:
            # Attempt to push tags into OpenMetadata for each parsed pair
            applied = await self.openmetadata.apply_defective_tags(catalog, parsed_pairs, reason=request.reason)

            # Also ensure the runtime mask table is updated immediately
            current_pairs = {
                (mask.layer_index, mask.head_index)
                for mask in self.inference.get_masked_heads()
            }
            current_pairs.update(parsed_pairs)
            masked_heads = await self.inference.set_masked_heads(current_pairs)

            parsed_heads = [
                HeadMask(
                    layer_index=layer,
                    layer_name=f"Layer_{layer + 1}",
                    head_index=head,
                    head_name=f"Head_{head + 1}",
                    reason=request.reason,
                )
                for (layer, head) in sorted(parsed_pairs)
            ]

            return OpenMetadataWebhookResponse(
                applied=bool(applied),
                parsed_heads=parsed_heads,
                masked_heads=masked_heads,
                reason=(
                    "Quarantine tags applied to OpenMetadata and local mask updated."
                    if applied
                    else "No tags were applied in OpenMetadata."
                ),
            )
        except Exception as exc:
            return OpenMetadataWebhookResponse(
                applied=False,
                parsed_heads=[],
                masked_heads=self.inference.get_masked_heads(),
                reason=str(exc),
            )

    async def apply_openmetadata_webhook(
        self,
        payload: dict[str, Any],
    ) -> OpenMetadataWebhookResponse:
        topology, _ = await self.ensure_topology()
        if not _payload_has_governance_tag(payload):
            return OpenMetadataWebhookResponse(
                applied=False,
                masked_heads=self.inference.get_masked_heads(),
                reason="Webhook ignored because it did not include a DEFECTIVE or QUARANTINED tag.",
            )

        parsed_pairs = _extract_webhook_head_pairs(payload, topology)
        if not parsed_pairs:
            return OpenMetadataWebhookResponse(
                applied=False,
                masked_heads=self.inference.get_masked_heads(),
                reason=(
                    "Webhook had a quarantine tag, but no Layer_N/Head_N reference could be parsed "
                    "from the payload."
                ),
            )

        current_pairs = {
            (mask.layer_index, mask.head_index)
            for mask in self.inference.get_masked_heads()
        }
        current_pairs.update(parsed_pairs)
        masked_heads = await self.inference.set_masked_heads(current_pairs)
        self._last_defect_sync_at = _utcnow()
        self._openmetadata_connected = True
        self._last_ingest_error = None

        parsed_heads = [
            mask for mask in masked_heads
            if (mask.layer_index, mask.head_index) in parsed_pairs
        ]
        return OpenMetadataWebhookResponse(
            applied=True,
            parsed_heads=parsed_heads,
            masked_heads=masked_heads,
            reason="OpenMetadata governance event applied to the live PyTorch head routing table.",
        )

    async def quarantine_circuit_stream(self, request: QuarantineRequest) -> StreamingResponse:
        topology, catalog = await self.ensure_topology()

        event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        async def producer() -> None:
            try:
                if not self.openmetadata_settings.openmetadata_enabled or catalog is None:
                    await event_queue.put(("error", {"message": "OpenMetadata disabled or catalog unavailable."}))
                    await event_queue.put(("close", {}))
                    return

                parsed_pairs: list[tuple[int, int]] = sorted({(h.layer_index, h.head_index) for h in request.heads})
                total = len(parsed_pairs)
                await event_queue.put(("progress", {"message": "quarantine_start", "total": total}))

                applied: list[HeadMask] = []
                idx = 0
                for (layer, head) in parsed_pairs:
                    idx += 1
                    await event_queue.put(("progress", {"message": "quarantine_pair_start", "pair": (layer, head), "idx": idx, "total": total}))
                    try:
                        # Apply tags for this single pair so we can stream per-head progress
                        applied_for_pair = await self.openmetadata.apply_defective_tags(catalog, {(layer, head)}, reason=request.reason)
                        if applied_for_pair:
                            applied.extend(applied_for_pair)
                        await event_queue.put(("progress", {"message": "quarantine_pair_done", "pair": (layer, head), "idx": idx, "applied_count": len(applied_for_pair)}))
                    except Exception as exc:
                        LOGGER.exception("Failed to apply defective tag for pair %s", (layer, head))
                        await event_queue.put(("progress", {"message": "quarantine_pair_error", "pair": (layer, head), "idx": idx, "error": str(exc)}))

                # Also ensure the runtime mask table is updated immediately
                current_pairs = {(mask.layer_index, mask.head_index) for mask in self.inference.get_masked_heads()}
                current_pairs.update(parsed_pairs)
                masked_heads = await self.inference.set_masked_heads(current_pairs)

                parsed_heads = [
                    HeadMask(
                        layer_index=layer,
                        layer_name=f"Layer_{layer + 1}",
                        head_index=head,
                        head_name=f"Head_{head + 1}",
                        reason=request.reason,
                    )
                    for (layer, head) in parsed_pairs
                ]

                await event_queue.put(("done", {"applied": len(applied) > 0, "parsed_heads": [p.model_dump(mode="json") for p in parsed_heads], "masked_heads": [m.model_dump(mode="json") for m in masked_heads], "reason": ("Quarantine tags applied." if applied else "No tags applied.")}))
            except Exception as exc:
                LOGGER.exception("quarantine_circuit_stream producer failed")
                await event_queue.put(("error", {"message": str(exc)}))
            finally:
                await event_queue.put(("close", {}))

        async def event_iterator() -> Any:
            producer_task = asyncio.create_task(producer())
            try:
                yield _format_sse("session", {"topology": topology.model_dump(mode="json") if topology is not None else None})
                while True:
                    event_name, payload = await event_queue.get()
                    if event_name == "close":
                        break
                    yield _format_sse(event_name, payload)
            finally:
                with suppress(asyncio.CancelledError):
                    await producer_task

        return StreamingResponse(event_iterator(), media_type="text/event-stream")

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
            return
        except Exception as exc:
            self._last_ingest_error = str(exc)
            self._openmetadata_connected = False
            LOGGER.exception("OpenMetadata lineage ingestion failed on first attempt.")

        # Retry once: attempt to refresh topology/catalog and replay the ingest.
        try:
            LOGGER.info("Attempting OpenMetadata catalog refresh and retrying lineage ingest.")
            topology, refreshed_catalog = await self.ensure_topology()
            if refreshed_catalog is not None:
                await self.openmetadata.ingest_step(refreshed_catalog, session_id, prompt, step)
                self._openmetadata_connected = True
                self._last_ingest_error = None
                LOGGER.info("OpenMetadata lineage ingest succeeded after catalog refresh.")
                return
        except Exception:
            LOGGER.exception("OpenMetadata lineage ingestion retry failed.")

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


@app.get("/api/v1/openmetadata/inspect")
async def inspect_openmetadata() -> dict[str, Any]:
    """Return debug payloads from OpenMetadata for the prompt table.

    Useful to verify that the catalog exists and to inspect tags/columns.
    """
    topology, catalog = await runtime.ensure_topology()
    if not runtime.openmetadata_settings.openmetadata_enabled:
        raise HTTPException(status_code=400, detail="OpenMetadata is not enabled in the runtime settings.")

    if catalog is None:
        return {
            "catalog_ready": False,
            "openmetadata": runtime._openmetadata_status(catalog_ready=False).model_dump(mode="json"),
            "message": "Catalog not available; run /api/v1/openmetadata/bootstrap",
        }

    # Fetch the prompt table payload (columns, tags)
    payload = await runtime.openmetadata._fetch_table_payload(catalog.prompt_table.table_fqn)
    return {
        "catalog_ready": True,
        "prompt_table_fqn": catalog.prompt_table.table_fqn,
        "prompt_table_payload": payload,
        "openmetadata": runtime._openmetadata_status(catalog_ready=True).model_dump(mode="json"),
    }


@app.get("/api/v1/openmetadata/catalog-detail")
async def inspect_openmetadata_catalog_detail() -> dict[str, Any]:
    """Return detailed payloads for all layer tables in the catalog.

    Useful to verify column-level tags (DEFECTIVE) were applied after quarantine.
    """
    topology, catalog = await runtime.ensure_topology()
    if not runtime.openmetadata_settings.openmetadata_enabled:
        raise HTTPException(status_code=400, detail="OpenMetadata is not enabled in the runtime settings.")

    if catalog is None:
        return {
            "catalog_ready": False,
            "openmetadata": runtime._openmetadata_status(catalog_ready=False).model_dump(mode="json"),
            "message": "Catalog not available; run /api/v1/openmetadata/bootstrap",
        }

    results: list[dict[str, Any]] = []
    for layer in catalog.layer_tables:
        payload = await runtime.openmetadata._fetch_table_payload(layer.table_fqn)
        results.append(
            {
                "table_fqn": layer.table_fqn,
                "layer_index": layer.layer_index,
                "payload": payload,
            }
        )

    return {
        "catalog_ready": True,
        "layer_tables": results,
        "openmetadata": runtime._openmetadata_status(catalog_ready=True).model_dump(mode="json"),
    }


@app.get("/api/v1/openmetadata/table-payload")
async def fetch_table_payload(table_fqn: str, fields: str = "columns,tags") -> dict[str, Any]:
    """Fetch a table payload from OpenMetadata with selectable `fields`.

    Use `fields=lineage,columns,tags` to include lineage details when available.
    """
    if not runtime.openmetadata_settings.openmetadata_enabled:
        raise HTTPException(status_code=400, detail="OpenMetadata is not enabled in the runtime settings.")

    try:
        response = await runtime.openmetadata._request_with_managed_auth(
            "GET",
            f"/v1/tables/name/{quote(table_fqn, safe='')}",
            params={"fields": fields},
        )
        response.raise_for_status()
    except Exception as e:
        LOGGER.exception("Failed to fetch table payload (fields=%s) for %s", fields, table_fqn)
        raise HTTPException(status_code=500, detail=str(e))

    return response.json()


@app.get("/api/v1/openmetadata/lineage")
async def proxy_openmetadata_lineage(table_fqn: str, direction: str = "both") -> dict[str, Any]:
    """Proxy OpenMetadata lineage API for a table and return raw lineage JSON.

    Example: /api/v1/openmetadata/lineage?table_fqn=Synapse_Neural_Service.gpt2.Transformer_Graph.Layer_10
    """
    if not runtime.openmetadata_settings.openmetadata_enabled:
        raise HTTPException(status_code=400, detail="OpenMetadata is not enabled in the runtime settings.")

    topology, catalog = await runtime.ensure_topology()
    if catalog is None:
        raise HTTPException(status_code=404, detail="Catalog not available; run /api/v1/openmetadata/bootstrap")

    # Use the OpenMetadata HTTP client to call the lineage endpoint with managed auth
    # Try the table-specific lineage endpoint first, then fall back to the
    # generic query-style endpoint which some OM versions expect.
    last_error: Exception | None = None
    try:
        path = f"/v1/lineage/table/{quote(table_fqn, safe='')}"
        response = await runtime.openmetadata._request_with_managed_auth("GET", path, params={"direction": direction})
        if response.status_code < 400:
            return response.json()
        last_error = RuntimeError(f"Lineage table endpoint returned {response.status_code}")
    except Exception as exc:
        last_error = exc

    # Fallback: query-style endpoint
    # Try a couple of query parameter name variants for different OM versions.
    for param_name in ("fqn", "fullyQualifiedName"):
        try:
            path = f"/v1/lineage"
            response = await runtime.openmetadata._request_with_managed_auth(
                "GET",
                path,
                params={"entity": "table", param_name: table_fqn, "direction": direction},
            )
            if response.status_code < 400:
                return response.json()
            last_error = RuntimeError(f"Lineage query endpoint returned {response.status_code} for param {param_name}")
        except Exception as exc:
            last_error = exc
    LOGGER.exception("Failed to fetch lineage (both endpoints) from OpenMetadata for %s", table_fqn)
    raise HTTPException(status_code=500, detail=str(last_error))


@app.get("/api/v1/openmetadata/lineage-sdk")
async def proxy_openmetadata_lineage_sdk(table_fqn: str, up_depth: int = 1, down_depth: int = 1) -> dict[str, Any]:
    """Fetch lineage using the OpenMetadata SDK (authenticated client).

    This avoids REST compatibility issues by calling the SDK method
    `get_lineage_by_name` which returns a parsed lineage payload.
    """
    try:
        metadata = runtime.openmetadata._metadata_client()
        lineage = metadata.get_lineage_by_name("table", table_fqn, up_depth=up_depth, down_depth=down_depth)
        return lineage or {}
    except Exception as exc:
        LOGGER.exception("OpenMetadata SDK lineage fetch failed for %s", table_fqn)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/openmetadata/parsed-evidence")
async def parsed_evidence(table_fqn: str, up_depth: int = 1, down_depth: int = 1) -> dict[str, Any]:
    """Return parsed `SYNAPSE_META` JSON blobs from lineage edges returned by the SDK.

    This provides a dashboard-friendly payload that surfaces causal evidence even
    when the OpenMetadata web UI hasn't indexed or displayed the edges yet.
    """
    if not runtime.openmetadata_settings.openmetadata_enabled:
        raise HTTPException(status_code=400, detail="OpenMetadata is not enabled in the runtime settings.")

    try:
        metadata = runtime.openmetadata._metadata_client()
        lineage = metadata.get_lineage_by_name("table", table_fqn, up_depth=up_depth, down_depth=down_depth)
    except Exception as exc:
        LOGGER.exception("OpenMetadata SDK lineage fetch failed for parsed-evidence %s", table_fqn)
        raise HTTPException(status_code=500, detail=str(exc))

    # Build id->fqn mapping when available to make results human-friendly
    id_to_fqn: dict[str, str] = {}
    try:
        # Map node id -> fullyQualifiedName
        for node in (lineage.get("nodes") or []):
            nid = node.get("id")
            if nid:
                id_to_fqn[str(nid)] = node.get("fullyQualifiedName") or node.get("name") or str(nid)
        # Also map the root entity if present
        root_entity = lineage.get("entity")
        if isinstance(root_entity, dict):
            rid = root_entity.get("id")
            if rid:
                id_to_fqn[str(rid)] = root_entity.get("fullyQualifiedName") or root_entity.get("name") or str(rid)
    except Exception:
        pass

    def _collect_edges(obj: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        if isinstance(obj, dict):
            # Check for an inline lineageDetails block
            ld = None
            if "lineageDetails" in obj and isinstance(obj.get("lineageDetails"), dict):
                ld = obj.get("lineageDetails")
                parent = obj
            elif "edge" in obj and isinstance(obj.get("edge"), dict) and isinstance(obj["edge"].get("lineageDetails"), dict):
                ld = obj["edge"].get("lineageDetails")
                parent = obj["edge"]

            if ld is not None:
                sql = ld.get("sqlQuery")
                columns_lineage = ld.get("columnsLineage")
                # Attempt to extract simple from/to identifiers when present
                from_ent = None
                to_ent = None
                if isinstance(parent.get("fromEntity"), dict):
                    raw_from = parent.get("fromEntity").get("fullyQualifiedName") or parent.get("fromEntity").get("id")
                    from_ent = id_to_fqn.get(str(raw_from), raw_from)
                elif isinstance(parent.get("fromEntity"), (str, int)):
                    raw_from = parent.get("fromEntity")
                    from_ent = id_to_fqn.get(str(raw_from), raw_from)

                if isinstance(parent.get("toEntity"), dict):
                    raw_to = parent.get("toEntity").get("fullyQualifiedName") or parent.get("toEntity").get("id")
                    to_ent = id_to_fqn.get(str(raw_to), raw_to)
                elif isinstance(parent.get("toEntity"), (str, int)):
                    raw_to = parent.get("toEntity")
                    to_ent = id_to_fqn.get(str(raw_to), raw_to)

                parsed_meta = None
                if isinstance(sql, str):
                    import re, json as _json

                    m = re.search(r"/\*\s*SYNAPSE_META:\s*(\{.*?\})\s*\*/", sql, flags=re.DOTALL)
                    if m:
                        try:
                            parsed_meta = _json.loads(m.group(1))
                        except Exception:
                            parsed_meta = {"_parse_error": "invalid json", "raw": m.group(1)}

                results.append(
                    {
                        "from": from_ent,
                        "to": to_ent,
                        "description": parent.get("description") or None,
                        "sqlQuery": sql,
                        "columnsLineage": columns_lineage,
                        "synapse_meta": parsed_meta,
                    }
                )

            for v in obj.values():
                results.extend(_collect_edges(v))

        elif isinstance(obj, list):
            for item in obj:
                results.extend(_collect_edges(item))

        return results

    edges = _collect_edges(lineage)
    return {"table_fqn": table_fqn, "edges": edges, "raw_lineage": lineage}


@app.post("/api/v1/openmetadata/mark-lineage-processed")
async def mark_lineage_processed(table_fqns: list[str]) -> dict[str, Any]:
    """Best-effort attempts to mark lineage processed and trigger reindexing in OpenMetadata.

    This endpoint tries several strategies (SDK helpers if present, REST PATCH on table,
    and reindex-like REST calls) and returns a diagnostic report so you can see what
    succeeded against your OpenMetadata deployment.
    """
    if not runtime.openmetadata_settings.openmetadata_enabled:
        raise HTTPException(status_code=400, detail="OpenMetadata is not enabled in the runtime settings.")

    results: list[dict[str, Any]] = []
    metadata = runtime.openmetadata._metadata_client()

    for table_fqn in table_fqns:
        entry: dict[str, Any] = {"table_fqn": table_fqn, "attempts": []}
        success = False

        # Strategy A: call common SDK helper names if available (best-effort)
        sdk_candidates = [
            "patch_lineage_processed_flag",
            "patch_lineage_processed",
            "patch_lineage",
            "patchProcessedLineage",
            "patch_lineage_processed_flag_by_name",
        ]
        for name in sdk_candidates:
            fn = getattr(metadata, name, None)
            if callable(fn):
                # Try several common arities for SDK helper methods (best-effort)
                tried = False
                for candidate_args in (("table", table_fqn, True), ("table", table_fqn), (table_fqn, True), (table_fqn,)):
                    try:
                        await asyncio.to_thread(fn, *candidate_args)
                        entry["attempts"].append({"method": name, "status": "ok", "args": candidate_args})
                        success = True
                        tried = True
                        break
                    except TypeError as exc:
                        # Signature mismatch; record and try next arity
                        entry["attempts"].append({"method": name, "error": str(exc), "args": candidate_args})
                        tried = True
                        continue
                    except Exception as exc:
                        entry["attempts"].append({"method": name, "error": str(exc), "args": candidate_args})
                        tried = True
                        break
                if tried and success:
                    break

        # Strategy B: PATCH the table entity with processedLineage flag (some OM versions accept this)
        if not success:
            try:
                resp = await runtime.openmetadata._request_with_managed_auth(
                    "PATCH",
                    f"/v1/tables/name/{quote(table_fqn, safe='')}",
                    json={"processedLineage": True},
                )
                if resp.status_code < 400:
                    entry["attempts"].append({"method": "rest_patch_table", "status": "ok", "http_status": resp.status_code})
                    success = True
                else:
                    entry["attempts"].append({"method": "rest_patch_table", "status": "error", "http_status": resp.status_code, "body": (resp.text or '')[:800]})
            except Exception as exc:
                entry["attempts"].append({"method": "rest_patch_table", "error": str(exc)})

        # Strategy C: Try to trigger a reindex/search update (best-effort)
        try:
            resp = await runtime.openmetadata._request_with_managed_auth("POST", "/v1/search/index", json={"query": table_fqn})
            if resp.status_code < 400:
                entry["attempts"].append({"method": "reindex_search_index", "status": "ok", "http_status": resp.status_code})
            else:
                entry["attempts"].append({"method": "reindex_search_index", "status": "error", "http_status": resp.status_code, "body": (resp.text or '')[:800]})
        except Exception as exc:
            entry["attempts"].append({"method": "reindex_search_index", "error": str(exc)})

        # Strategy D: POST to lineage endpoint to ask the server to process (some versions accept different shapes)
        try:
            resp = await runtime.openmetadata._request_with_managed_auth("POST", "/v1/lineage", json={"entity": "table", "fqn": table_fqn, "processed": True})
            if resp.status_code < 400:
                entry["attempts"].append({"method": "post_lineage_processed", "status": "ok", "http_status": resp.status_code})
                success = True
            else:
                entry["attempts"].append({"method": "post_lineage_processed", "status": "error", "http_status": resp.status_code, "body": (resp.text or '')[:800]})
        except Exception as exc:
            entry["attempts"].append({"method": "post_lineage_processed", "error": str(exc)})

        entry["success"] = success
        results.append(entry)

    return {"results": results}


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


@app.post("/api/v1/autopsy/causal", response_model=CausalAutopsyResponse)
async def run_causal_autopsy(request: CausalAutopsyRequest) -> CausalAutopsyResponse:
    return await runtime.run_causal_autopsy(request)


@app.post("/api/v1/autopsy/discover_circuit", response_model=CircuitDiscoveryResponse)
async def discover_circuit(request: CircuitDiscoveryRequest) -> CircuitDiscoveryResponse:
    return await runtime.discover_circuit(request)


@app.post("/api/v1/autopsy/discover_circuit/stream")
async def discover_circuit_stream(request: CircuitDiscoveryRequest) -> StreamingResponse:
    return await runtime.discover_circuit_stream(request)


@app.post("/api/v1/webhooks/openmetadata", response_model=OpenMetadataWebhookResponse)
async def openmetadata_webhook(request: Request) -> OpenMetadataWebhookResponse:
    # Validate optional webhook secret header
    secret_header = request.headers.get("X-OpenMetadata-Secret")
    configured = runtime.openmetadata_settings.openmetadata_webhook_secret
    if configured:
        if not secret_header or secret_header != configured:
            raise HTTPException(status_code=403, detail="Invalid webhook secret")

    payload = await request.json()
    return await runtime.apply_openmetadata_webhook(payload)


@app.post("/api/v1/openmetadata/quarantine", response_model=OpenMetadataWebhookResponse)
async def quarantine_openmetadata(request: QuarantineRequest) -> OpenMetadataWebhookResponse:
    return await runtime.quarantine_circuit(request)


@app.post("/api/v1/openmetadata/quarantine/stream")
async def quarantine_openmetadata_stream(request: QuarantineRequest) -> StreamingResponse:
    return await runtime.quarantine_circuit_stream(request)


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


def _select_autopsy_target(
    trace: AttentionTrace,
    *,
    layer_index: int | None,
    head_index: int | None,
) -> CausalAutopsyTarget:
    if layer_index is not None and head_index is not None:
        return CausalAutopsyTarget(
            layer_index=layer_index,
            layer_name=f"Layer_{layer_index + 1}",
            head_index=head_index,
            head_name=f"Head_{head_index + 1}",
            selection_reason="User-selected target head.",
        )

    candidates: list[tuple[float, int, str, int, str]] = []
    for step in trace.steps:
        for layer in step.layers:
            if layer_index is not None and layer.layer_index != layer_index:
                continue
            for head in layer.top_heads:
                if head_index is not None and head.head_index != head_index:
                    continue
                candidates.append(
                    (
                        head.max_attention_score,
                        layer.layer_index,
                        layer.layer_name,
                        head.head_index,
                        head.head_name,
                    )
                )

    if not candidates:
        raise RuntimeError("No traced head was available to ablate. Run faithful tracing with a valid HF model.")

    _, selected_layer_index, selected_layer_name, selected_head_index, selected_head_name = max(
        candidates,
        key=lambda item: item[0],
    )
    return CausalAutopsyTarget(
        layer_index=selected_layer_index,
        layer_name=selected_layer_name,
        head_index=selected_head_index,
        head_name=selected_head_name,
        selection_reason="Automatically selected highest-attention head from the baseline trace.",
    )


def _causal_verdict(causal_effect_score: float) -> str:
    if causal_effect_score >= 0.5:
        return "Strong causal effect: output changed substantially under head ablation."
    if causal_effect_score >= 0.2:
        return "Moderate causal effect: output changed measurably under head ablation."
    if causal_effect_score >= 0.05:
        return "Weak causal effect: output changed slightly under head ablation."
    return "No meaningful causal effect detected for this deterministic replay."


def _rank_circuit_heads(trace: AttentionTrace, *, top_k: int) -> list[CircuitHead]:
    scored: dict[tuple[int, int], CircuitHead] = {}
    for step in trace.steps:
        for layer in step.layers:
            for head in layer.top_heads:
                key = (layer.layer_index, head.head_index)
                score = max(head.max_attention_score, head.mean_attention_score, head.l2_norm)
                current = scored.get(key)
                if current is None:
                    scored[key] = CircuitHead(
                        layer_index=layer.layer_index,
                        layer_name=layer.layer_name,
                        head_index=head.head_index,
                        head_name=head.head_name,
                        activation_score=round(score, 6),
                    )
                    continue
                scored[key] = current.model_copy(
                    update={"activation_score": round(max(current.activation_score, score), 6)}
                )

    ranked = sorted(
        scored.values(),
        key=lambda head: head.activation_score,
        reverse=True,
    )
    if not ranked:
        raise RuntimeError(
            "No active heads were captured. Run faithful tracing with a valid Hugging Face model."
        )
    return ranked[:top_k]


def _count_target_token(text: str, target: str) -> int:
    target = target.strip()
    if not target:
        return 0

    if re.fullmatch(r"\w+", target):
        return len(re.findall(rf"\b{re.escape(target)}\b", text, flags=re.IGNORECASE))
    return text.lower().count(target.lower())


def _target_reduction_score(*, baseline_target_count: int, ablated_target_count: int) -> float:
    if baseline_target_count <= 0:
        return 0.0
    reduced_by = max(0, baseline_target_count - ablated_target_count)
    return reduced_by / baseline_target_count


def _select_best_circuit_result(
    sweep_results: list[CircuitAblationResult],
    *,
    baseline_target_count: int,
) -> CircuitAblationResult | None:
    if not sweep_results:
        return None

    def score(result: CircuitAblationResult) -> tuple[float, float, int]:
        target_reduction = _target_reduction_score(
            baseline_target_count=baseline_target_count,
            ablated_target_count=result.target_count,
        )
        # Prefer circuits that remove the target token, then larger causal deltas,
        # then smaller circuits for easier governance.
        return (target_reduction, result.causal_effect_score, -len(result.masked_heads))

    return max(sweep_results, key=score)


def _circuit_verdict(
    *,
    baseline_target_count: int,
    best_result: CircuitAblationResult | None,
    target_hallucination_token: str,
) -> str:
    if baseline_target_count <= 0:
        return (
            f"Target token '{target_hallucination_token}' was not present in the "
            "deterministic baseline, "
            "so this run cannot prove a causal hallucination circuit for that token."
        )
    if best_result is None:
        return "No ablation sweep completed; no circuit could be discovered."
    if best_result.target_count == 0:
        return (
            "Circuit discovered: masking the returned head set removed the target token from the "
            "deterministic replay."
        )
    if best_result.causal_effect_score >= 0.2:
        return (
            "Partial circuit found: the target token survived, but the deterministic replay "
            "changed "
            "enough to justify deeper pair or triple-head sweeps."
        )
    return "No causal circuit was proven by the top-head single and pair ablation sweep."


def _payload_has_governance_tag(payload: dict[str, Any]) -> bool:
    payload_text = _json_text(payload).upper()
    return "DEFECTIVE" in payload_text or "QUARANTINED" in payload_text


def _extract_webhook_head_pairs(
    payload: dict[str, Any],
    topology: ModelTopology,
) -> set[tuple[int, int]]:
    payload_text = _json_text(payload)
    layer_numbers = _extract_number_set(payload_text, r"\blayer[_\s:./-]*(\d+)\b")
    head_numbers = _extract_number_set(payload_text, r"\bhead[_\s:./-]*(\d+)\b")

    pairs: set[tuple[int, int]] = set()
    if layer_numbers and head_numbers:
        for layer_number in layer_numbers:
            for head_number in head_numbers:
                layer_index = layer_number - 1
                head_index = head_number - 1
                if _topology_has_head(topology, layer_index, head_index):
                    pairs.add((layer_index, head_index))
        return pairs

    if layer_numbers:
        for layer_number in layer_numbers:
            layer_index = layer_number - 1
            layer = _topology_layer(topology, layer_index)
            if layer is None:
                continue
            for head_index in range(layer.head_count):
                pairs.add((layer_index, head_index))
    return pairs


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _extract_number_set(text: str, pattern: str) -> set[int]:
    values: set[int] = set()
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.add(value)
    return values


def _topology_layer(topology: ModelTopology, layer_index: int) -> Any | None:
    for layer in topology.layers:
        if layer.layer_index == layer_index:
            return layer
    return None


def _topology_has_head(topology: ModelTopology, layer_index: int, head_index: int) -> bool:
    layer = _topology_layer(topology, layer_index)
    return layer is not None and 0 <= head_index < layer.head_count
