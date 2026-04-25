from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import os
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
import difflib
from contextlib import suppress
from enum import StrEnum
from threading import Lock
from typing import Any

import httpx
import torch
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)

_LAYER_PATTERNS = (
    re.compile(r"layers\.(\d+)\."),
    re.compile(r"h\.(\d+)\."),
    re.compile(r"blocks\.(\d+)\."),
)


class GenerationBackend(StrEnum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class AnalysisMode(StrEnum):
    INLINE = "inline"
    SHADOW = "shadow"


class TraceExecutionMode(StrEnum):
    AUTO = "auto"
    FAST = "fast"
    FAITHFUL = "faithful"


class TraceFidelity(StrEnum):
    EXACT = "exact"
    PROXY = "proxy"


class EvidenceQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    label: str
    exactness: str
    causal_validation: str
    black_box_gaps: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    max_new_tokens: int = Field(default=160, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)
    stream: bool = True
    execution_mode: TraceExecutionMode = TraceExecutionMode.AUTO
    trace_model_name: str | None = Field(
        default=None,
        description="Optional Hugging Face model id to use for hook-based tracing.",
    )


class HeadMask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int = Field(ge=0)
    layer_name: str
    head_index: int = Field(ge=0)
    head_name: str
    reason: str | None = None


class LayerTopology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int = Field(ge=0)
    layer_name: str
    head_count: int = Field(ge=1)
    head_dim: int = Field(ge=1)


class ModelTopology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    device: str
    total_layers: int = Field(ge=0)
    total_heads: int = Field(ge=0)
    layers: list[LayerTopology] = Field(default_factory=list)


class HeadActivation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    head_index: int = Field(ge=0)
    head_name: str
    masked: bool = False
    max_attention_score: float
    mean_attention_score: float
    l2_norm: float
    top_source_positions: list[int] = Field(default_factory=list)
    top_source_tokens: list[str] = Field(default_factory=list)
    raw_last_token_attention: list[float] = Field(default_factory=list)


class LayerActivation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int = Field(ge=0)
    layer_name: str
    sequence_length: int = Field(ge=1)
    head_count: int = Field(ge=1)
    masked_head_names: list[str] = Field(default_factory=list)
    dominant_source_tokens: list[str] = Field(default_factory=list)
    top_heads: list[HeadActivation] = Field(default_factory=list)
    full_last_token_attention_matrix: list[list[float]] | None = None


class TokenStepCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=0)
    generated_token: str
    generated_token_id: int = Field(ge=0)
    prompt_plus_generation_length: int = Field(ge=1)
    masked_heads: list[str] = Field(default_factory=list)
    layers: list[LayerActivation] = Field(default_factory=list)
    high_activation_path: list[str] = Field(default_factory=list)
    evidence_tokens: list[str] = Field(default_factory=list)
    explanation: str | None = None


class TraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str
    dominant_layers: list[str] = Field(default_factory=list)
    dominant_heads: list[str] = Field(default_factory=list)
    influential_tokens: list[str] = Field(default_factory=list)
    masked_heads_applied: list[str] = Field(default_factory=list)


class AttentionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_prompt: str
    generation_model: str
    analysis_model: str
    generation_backend: GenerationBackend
    analysis_mode: AnalysisMode
    trace_fidelity: TraceFidelity
    prompt_token_count: int = Field(ge=1)
    generated_text: str = ""
    analysis_error: str | None = None
    match_score: float | None = None
    fidelity_reason: str | None = None
    evidence_quality: EvidenceQuality | None = None
    summary: TraceSummary | None = None
    steps: list[TokenStepCapture] = Field(default_factory=list)


class GenerationChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: GenerationBackend
    model: str
    token: str = ""
    done: bool = False
    trace_step: TokenStepCapture | None = None
    trace: AttentionTrace | None = None


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: GenerationBackend
    generation_model: str
    analysis_model: str
    text: str
    trace: AttentionTrace


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYNAPSE_",
        env_file=[".env", "backend/.env", "../.env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    use_ollama_if_available: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_health_path: str = "/api/version"
    ollama_model: str = "qwen2.5:3b-instruct"
    ollama_request_timeout_seconds: float = 120.0
    ollama_health_timeout_seconds: float = 2.0

    hf_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_revision: str | None = None
    hf_token: str | None = None
    hf_local_files_only: bool = False
    hf_device: str | None = None
    hf_dtype: str | None = None
    hf_trust_remote_code: bool = False
    hf_attention_implementation: str = "eager"
    hf_enable_chat_template: bool = True

    capture_top_k_heads: int = 6
    capture_top_k_positions: int = 8
    capture_context_window: int = 128
    capture_activation_threshold: float = 0.0
    capture_full_attention_matrix: bool = False

    preload_shadow_model: bool = False
    shadow_max_new_tokens: int | None = None


StepListener = Callable[[TokenStepCapture], Awaitable[None] | None]


class HeadMaskStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._masked_heads: set[tuple[int, int]] = set()

    def replace(self, masked_heads: set[tuple[int, int]]) -> None:
        with self._lock:
            self._masked_heads = set(masked_heads)

    def snapshot(self) -> set[tuple[int, int]]:
        with self._lock:
            return set(self._masked_heads)

    def snapshot_grouped(self) -> dict[int, set[int]]:
        return _group_mask_heads(self.snapshot())

    def snapshot_models(self) -> list[HeadMask]:
        masked_models = [
            HeadMask(
                layer_index=layer_index,
                layer_name=f"Layer_{layer_index + 1}",
                head_index=head_index,
                head_name=f"Head_{head_index + 1}",
            )
            for layer_index, head_index in sorted(self.snapshot())
        ]
        return masked_models


class OllamaClient:
    def __init__(self, settings: InferenceSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_request_timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def is_available(self) -> bool:
        try:
            response = await self._client.get(
                self._settings.ollama_health_path,
                timeout=self._settings.ollama_health_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        payload = {
            "model": self._settings.ollama_model,
            "prompt": request.prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_new_tokens,
                "stop": request.stop,
            },
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        async with self._client.stream("POST", "/api/generate", json=payload) as response:
            if response.status_code >= 400:
                body_bytes = await response.aread()
                body_text = body_bytes.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Ollama generation failed ({response.status_code}): {body_text.strip()}"
                )

            async for line in response.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)
                token = chunk.get("response", "")
                done = bool(chunk.get("done", False))

                if token:
                    yield GenerationChunk(
                        backend=GenerationBackend.OLLAMA,
                        model=self._settings.ollama_model,
                        token=token,
                    )

                if done:
                    return


class _TraceSession:
    def __init__(self, context_window: int, masked_heads_by_layer: dict[int, set[int]]) -> None:
        self.context_window = context_window
        self.masked_heads_by_layer = masked_heads_by_layer
        self.current_layer_slices: dict[int, Tensor] = {}


class HookedTransformerRunner:
    def __init__(self, settings: InferenceSettings) -> None:
        self._settings = settings
        self._device = settings.hf_device or _resolve_device()
        self._dtype = _resolve_dtype(settings.hf_dtype, self._device)
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None
        self._load_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()
        self._hook_handles: list[Any] = []
        self._active_trace_session: _TraceSession | None = None
        self._mask_store = HeadMaskStore()
        self._model_topology: ModelTopology | None = None

    async def ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        async with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            await asyncio.to_thread(self._load_sync)

    async def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._model = None
        self._tokenizer = None
        self._model_topology = None

    async def get_model_topology(self) -> ModelTopology:
        await self.ensure_loaded()
        if self._model_topology is None:
            raise RuntimeError("Model topology is unavailable even though the model is loaded.")
        return self._model_topology

    async def replace_masked_heads(self, masked_heads: set[tuple[int, int]]) -> list[HeadMask]:
        self._mask_store.replace(masked_heads)
        return self._mask_store.snapshot_models()

    def get_masked_heads(self) -> list[HeadMask]:
        return self._mask_store.snapshot_models()

    def is_ready(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    async def use_model(self, model_name: str) -> None:
        normalized_name = model_name.strip()
        if not normalized_name or normalized_name == self._settings.hf_model_name:
            return

        async with self._generation_lock:
            async with self._load_lock:
                for handle in self._hook_handles:
                    handle.remove()
                self._hook_handles.clear()
                self._tokenizer = None
                self._model = None
                self._model_topology = None
                self._settings.hf_model_name = normalized_name

    async def capture_trace(
        self,
        request: GenerationRequest,
        *,
        analysis_mode: AnalysisMode,
        step_listener: StepListener | None = None,
        # Fractional match score between the live generation and the shadow HF run (0.0-1.0)
        match_score: float | None = None
        # Human readable reason describing why the trace was marked proxy/exact
    ) -> AttentionTrace:
        trace: AttentionTrace | None = None
        async for chunk in self.generate_stream(
            request,
            analysis_mode=analysis_mode,
            step_listener=step_listener,
        ):
            if chunk.trace is not None:
                trace = chunk.trace

        if trace is None:
            raise RuntimeError("The hook runner completed without a final trace payload.")
        return trace

    async def generate_stream(
        self,
        request: GenerationRequest,
        *,
        analysis_mode: AnalysisMode,
        step_listener: StepListener | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        await self.ensure_loaded()
        tokenizer = self._require_tokenizer()
        model = self._require_model()

        prompt_text = self._render_prompt(request.prompt, request.system_prompt)
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        prompt_token_count = int(input_ids.shape[1])

        trace = AttentionTrace(
            source_prompt=request.prompt,
            generation_model=self._settings.hf_model_name,
            analysis_model=self._settings.hf_model_name,
            generation_backend=GenerationBackend.HUGGINGFACE,
            analysis_mode=analysis_mode,
            trace_fidelity=TraceFidelity.EXACT,
            prompt_token_count=prompt_token_count,
        )

        generated_text_parts: list[str] = []
        eos_token_id = tokenizer.eos_token_id

        async with self._generation_lock:
            self._active_trace_session = _TraceSession(
                context_window=self._settings.capture_context_window,
                masked_heads_by_layer=self._mask_store.snapshot_grouped(),
            )
            try:
                for step_index in range(request.max_new_tokens):
                    self._active_trace_session.current_layer_slices.clear()
                    outputs = await asyncio.to_thread(
                        self._forward_sync,
                        model,
                        input_ids,
                        attention_mask,
                    )

                    next_token_id = self._sample_next_token(
                        outputs.logits[:, -1, :],
                        request.temperature,
                        request.top_p,
                    )

                    is_eos = eos_token_id is not None and next_token_id == eos_token_id
                    token_text = "" if is_eos else tokenizer.decode(
                        [next_token_id],
                        skip_special_tokens=False,
                    )

                    if token_text:
                        generated_text_parts.append(token_text)
                        trace.generated_text += token_text

                    step_capture = self._build_step_capture(
                        step_index=step_index,
                        generated_token=token_text,
                        generated_token_id=next_token_id,
                        prompt_plus_generation_length=prompt_token_count + step_index + 1,
                        context_tokens=_format_token_window(
                            tokenizer,
                            input_ids[0].detach().to(device="cpu").tolist(),
                            self._settings.capture_context_window,
                        ),
                    )
                    trace.steps.append(step_capture)

                    if step_listener is not None:
                        callback_result = step_listener(step_capture)
                        if inspect.isawaitable(callback_result):
                            await callback_result

                    if token_text:
                        yield GenerationChunk(
                            backend=GenerationBackend.HUGGINGFACE,
                            model=self._settings.hf_model_name,
                            token=token_text,
                            trace_step=step_capture,
                        )

                    next_token_tensor = torch.tensor(
                        [[next_token_id]],
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    input_ids = torch.cat((input_ids, next_token_tensor), dim=-1)
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype),
                        ),
                        dim=-1,
                    )

                    if is_eos or _matches_any_stop_sequence("".join(generated_text_parts), request.stop):
                        break

                trace = trace.model_copy(update={"summary": self._summarize_trace(trace)})
                trace = trace.model_copy(update={"evidence_quality": _assess_trace_evidence(trace)})
                yield GenerationChunk(
                    backend=GenerationBackend.HUGGINGFACE,
                    model=self._settings.hf_model_name,
                    done=True,
                    trace=trace,
                )
            finally:
                self._active_trace_session = None

    def _load_sync(self) -> None:
        tokenizer_load_kwargs: dict[str, Any] = {
            "trust_remote_code": self._settings.hf_trust_remote_code,
            "local_files_only": self._settings.hf_local_files_only,
        }
        if self._settings.hf_revision:
            tokenizer_load_kwargs["revision"] = self._settings.hf_revision

        # Normalize HF model name. If the configured value looks like an Ollama
        # identifier (e.g. 'phi3:latest'), skip attempting to contact the Hugging
        # Face Hub and treat the HF model as unavailable to avoid spurious 401s
        # when the user meant to use Ollama. Otherwise strip simple colon tags
        # (e.g. 'model:tag') for hub operations.
        hf_name_raw = self._settings.hf_model_name or ""
        hf_name_to_load = hf_name_raw
        if ":" in hf_name_raw:
            # Heuristic: if there's no '/' and the suffix looks like a simple tag
            # (common for Ollama identifiers like 'name:latest'), assume this is
            # NOT a Hugging Face repo id and skip HF loading.
            prefix, suffix = hf_name_raw.split(":", 1)
            if "/" not in hf_name_raw and re.match(r"^[\w.\-]+$", suffix):
                LOGGER.warning(
                    "Configured HF model '%s' looks like a non-HF identifier (e.g. Ollama). Skipping Hugging Face load. "
                    "Set 'ollama_model' to use Ollama or set a valid HF repo id.",
                    hf_name_raw,
                )
                self._tokenizer = None
                self._model = None
                self._model_topology = ModelTopology(
                    model_name=hf_name_raw,
                    device="unavailable",
                    total_layers=0,
                    total_heads=0,
                    layers=[],
                )
                return
            # Otherwise treat the part before ':' as the repo id and warn.
            hf_name_to_load = hf_name_raw.split(":", 1)[0]
            LOGGER.warning(
                "HuggingFace model name '%s' contains ':'; using '%s' for hub operations.",
                hf_name_raw,
                hf_name_to_load,
            )

        # Allow an explicit HF token via settings or common env vars so private
        # repositories can be accessed when present.
        hf_token = (
            self._settings.hf_token
            or os.environ.get("HF_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        if hf_token:
            tokenizer_load_kwargs["use_auth_token"] = hf_token

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                hf_name_to_load,
                **tokenizer_load_kwargs,
            )
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model_load_kwargs: dict[str, Any] = {
                "torch_dtype": self._dtype,
                "trust_remote_code": self._settings.hf_trust_remote_code,
                "local_files_only": self._settings.hf_local_files_only,
                "low_cpu_mem_usage": True,
            }
            if hf_token:
                model_load_kwargs["use_auth_token"] = hf_token
            if self._settings.hf_revision:
                model_load_kwargs["revision"] = self._settings.hf_revision

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_name_to_load,
                    attn_implementation=self._settings.hf_attention_implementation,
                    **model_load_kwargs,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_name_to_load,
                    **model_load_kwargs,
                )

            model.to(self._device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._register_attention_hooks(model)
            self._model_topology = self._inspect_model_topology(model)
        except Exception as exc:
            LOGGER.exception("Failed to load HuggingFace model '%s': %s", self._settings.hf_model_name, exc)
            # Provide clearer guidance for common failures (auth / private repo / wrong identifier)
            exc_text = str(exc)
            if any(tok in exc_text for tok in ("401", "Unauthorized", "Repository Not Found", "is not a local folder")):
                LOGGER.error(
                    "HuggingFace access error. If the model is private, set the token via the SYNAPSE_HF_TOKEN env var or HF_HUB_TOKEN, or run 'huggingface-cli login'.\n"
                    "If you intended to use an Ollama identifier (e.g. 'phi3:latest'), set that as the Ollama model via the 'ollama_model' setting instead of 'hf_model_name'."
                )
            # Try a safe fallback to a small public model so the visualizer has a topology.
            fallback_candidate = "gpt2"
            if not self._settings.hf_local_files_only and self._settings.hf_model_name != fallback_candidate:
                try:
                    LOGGER.info("Attempting fallback HuggingFace model '%s'", fallback_candidate)
                    tokenizer = AutoTokenizer.from_pretrained(fallback_candidate)
                    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token

                    model_load_kwargs = {
                        "torch_dtype": self._dtype,
                        "trust_remote_code": self._settings.hf_trust_remote_code,
                        "local_files_only": False,
                        "low_cpu_mem_usage": True,
                    }
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            fallback_candidate,
                            attn_implementation=self._settings.hf_attention_implementation,
                            **model_load_kwargs,
                        )
                    except TypeError:
                        model = AutoModelForCausalLM.from_pretrained(fallback_candidate, **model_load_kwargs)

                    model.to(self._device)
                    model.eval()
                    self._tokenizer = tokenizer
                    self._model = model
                    self._register_attention_hooks(model)
                    self._model_topology = self._inspect_model_topology(model)
                    LOGGER.info("Fallback model '%s' loaded successfully.", fallback_candidate)
                    return
                except Exception as exc2:
                    LOGGER.exception("Fallback HuggingFace model '%s' failed: %s", fallback_candidate, exc2)

            # Final fallback: minimal topology so the runtime can continue without HF tracing.
            self._tokenizer = None
            self._model = None
            self._model_topology = ModelTopology(
                model_name=self._settings.hf_model_name,
                device="unavailable",
                total_layers=0,
                total_heads=0,
                layers=[],
            )

    def _register_attention_hooks(self, model: PreTrainedModel) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        for module_name, module in model.named_modules():
            if not _is_attention_module(module_name, module):
                continue
            layer_index = _extract_layer_index(module_name)
            if layer_index is None:
                continue
            handle = module.register_forward_hook(self._make_attention_hook(module_name))
            self._hook_handles.append(handle)
            projection_module = getattr(module, "o_proj", None) or getattr(module, "out_proj", None)
            if isinstance(projection_module, nn.Module):
                mask_handle = projection_module.register_forward_pre_hook(
                    self._make_projection_mask_hook(layer_index, module)
                )
                self._hook_handles.append(mask_handle)

    def _make_attention_hook(self, module_name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        layer_index = _extract_layer_index(module_name)

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if layer_index is None or self._active_trace_session is None:
                return
            attention_tensor = _extract_attention_tensor(output)
            if attention_tensor is None or attention_tensor.ndim != 4 or attention_tensor.shape[0] == 0:
                return

            detached = attention_tensor.detach().to(device="cpu", dtype=torch.float32)
            last_query = detached[0, :, -1, :]
            masked_heads = self._active_trace_session.masked_heads_by_layer.get(layer_index, set())
            for head_index in masked_heads:
                if 0 <= head_index < last_query.shape[0]:
                    last_query[head_index] = 0
            if last_query.shape[-1] > self._active_trace_session.context_window:
                last_query = last_query[:, -self._active_trace_session.context_window :]
            self._active_trace_session.current_layer_slices[layer_index] = last_query.contiguous()

        return hook

    def _make_projection_mask_hook(
        self,
        layer_index: int,
        attention_module: nn.Module,
    ) -> Callable[[nn.Module, tuple[Any, ...]], tuple[Any, ...] | None]:
        num_heads = int(getattr(attention_module, "num_heads", 0))
        head_dim = int(getattr(attention_module, "head_dim", 0))

        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if (
                self._active_trace_session is None
                or not inputs
                or num_heads <= 0
                or head_dim <= 0
            ):
                return None

            masked_heads = self._active_trace_session.masked_heads_by_layer.get(layer_index)
            if not masked_heads:
                return None

            hidden_states = inputs[0]
            if not isinstance(hidden_states, Tensor) or hidden_states.ndim != 3:
                return None

            masked_hidden_states = hidden_states.clone()
            expected_width = num_heads * head_dim
            if masked_hidden_states.shape[-1] < expected_width:
                return None

            for head_index in masked_heads:
                if not 0 <= head_index < num_heads:
                    continue
                start = head_index * head_dim
                end = start + head_dim
                masked_hidden_states[..., start:end] = 0

            return (masked_hidden_states, *inputs[1:])

        return hook

    def _forward_sync(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Any:
        with torch.inference_mode():
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

    def _sample_next_token(self, logits: Tensor, temperature: float, top_p: float) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        scaled_logits = logits / max(temperature, 1e-6)
        probabilities = torch.softmax(scaled_logits, dim=-1)

        if top_p < 1.0:
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
            sorted_mask = cumulative_probabilities > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_probabilities = sorted_probabilities.masked_fill(sorted_mask, 0.0)
            normalization = sorted_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            sorted_probabilities = sorted_probabilities / normalization
            sampled_index = torch.multinomial(sorted_probabilities, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_index)
            return int(next_token.item())

        return int(torch.multinomial(probabilities, num_samples=1).item())

    def _build_step_capture(
        self,
        *,
        step_index: int,
        generated_token: str,
        generated_token_id: int,
        prompt_plus_generation_length: int,
        context_tokens: list[str],
    ) -> TokenStepCapture:
        if self._active_trace_session is None:
            raise RuntimeError("The active trace session was unexpectedly missing while building a step.")

        layers: list[LayerActivation] = []
        high_activation_path: list[str] = []
        masked_head_names = _format_mask_names(self._active_trace_session.masked_heads_by_layer)

        for layer_index, head_matrix in sorted(self._active_trace_session.current_layer_slices.items()):
            layer_capture = self._summarize_layer(
                layer_index,
                head_matrix,
                context_tokens,
                self._active_trace_session.masked_heads_by_layer.get(layer_index, set()),
            )
            layers.append(layer_capture)

            if not layer_capture.top_heads:
                continue
            top_head = layer_capture.top_heads[0]
            if top_head.max_attention_score >= self._settings.capture_activation_threshold:
                high_activation_path.append(f"{layer_capture.layer_name}:{top_head.head_name}")

        ranked_layers = sorted(
            (layer for layer in layers if layer.top_heads),
            key=lambda layer: layer.top_heads[0].max_attention_score,
            reverse=True,
        )
        evidence_tokens = _dedupe_tokens(
            token
            for layer in ranked_layers[:3]
            for token in layer.dominant_source_tokens
        )[:6]

        return TokenStepCapture(
            step_index=step_index,
            generated_token=generated_token,
            generated_token_id=generated_token_id,
            prompt_plus_generation_length=prompt_plus_generation_length,
            masked_heads=masked_head_names,
            layers=layers,
            high_activation_path=high_activation_path,
            evidence_tokens=evidence_tokens,
            explanation=_build_step_explanation(
                generated_token=generated_token,
                ranked_layers=ranked_layers,
                evidence_tokens=evidence_tokens,
                masked_head_names=masked_head_names,
            ),
        )

    def _summarize_layer(
        self,
        layer_index: int,
        head_matrix: Tensor,
        context_tokens: list[str],
        masked_heads: set[int],
    ) -> LayerActivation:
        head_count, sequence_length = head_matrix.shape
        summaries: list[HeadActivation] = []

        for head_index in range(head_count):
            head_vector = head_matrix[head_index]
            top_k_positions = min(self._settings.capture_top_k_positions, sequence_length)
            top_positions = torch.topk(head_vector, k=top_k_positions).indices.tolist()
            summaries.append(
                HeadActivation(
                    head_index=head_index,
                    head_name=f"Head_{head_index + 1}",
                    masked=head_index in masked_heads,
                    max_attention_score=float(head_vector.max().item()),
                    mean_attention_score=float(head_vector.mean().item()),
                    l2_norm=float(torch.linalg.vector_norm(head_vector).item()),
                    top_source_positions=[int(position) for position in top_positions],
                    top_source_tokens=[
                        context_tokens[position]
                        if 0 <= position < len(context_tokens)
                        else f"pos:{position}"
                        for position in top_positions
                    ],
                    raw_last_token_attention=head_vector.tolist(),
                )
            )

        summaries.sort(
            key=lambda item: (item.max_attention_score, item.l2_norm, item.mean_attention_score),
            reverse=True,
        )

        layer_matrix = head_matrix.tolist() if self._settings.capture_full_attention_matrix else None
        top_heads = summaries[: self._settings.capture_top_k_heads]

        return LayerActivation(
            layer_index=layer_index,
            layer_name=f"Layer_{layer_index + 1}",
            sequence_length=sequence_length,
            head_count=head_count,
            masked_head_names=[f"Head_{head_index + 1}" for head_index in sorted(masked_heads)],
            dominant_source_tokens=_dedupe_tokens(
                token
                for head in top_heads[:3]
                for token in head.top_source_tokens
            )[:5],
            top_heads=top_heads,
            full_last_token_attention_matrix=layer_matrix,
        )

    def _summarize_trace(self, trace: AttentionTrace) -> TraceSummary:
        layer_counter: Counter[str] = Counter()
        head_counter: Counter[str] = Counter()
        token_counter: Counter[str] = Counter()

        for step in trace.steps:
            for path_entry in step.high_activation_path:
                if ":" not in path_entry:
                    continue
                layer_name, head_name = path_entry.split(":", 1)
                layer_counter[layer_name] += 1
                head_counter[f"{layer_name}:{head_name}"] += 1
            for token in step.evidence_tokens:
                if _is_informative_token(token):
                    token_counter[token] += 1

        dominant_layers = [name for name, _ in layer_counter.most_common(4)]
        dominant_heads = [name for name, _ in head_counter.most_common(5)]
        influential_tokens = [token for token, _ in token_counter.most_common(6)]
        masked_heads_applied = trace.steps[-1].masked_heads if trace.steps else []

        return TraceSummary(
            explanation=_build_trace_summary_explanation(
                trace.trace_fidelity,
                dominant_heads,
                influential_tokens,
                masked_heads_applied,
            ),
            dominant_layers=dominant_layers,
            dominant_heads=dominant_heads,
            influential_tokens=influential_tokens,
            masked_heads_applied=masked_heads_applied,
        )

    def _render_prompt(self, prompt: str, system_prompt: str | None) -> str:
        tokenizer = self._require_tokenizer()

        if self._settings.hf_enable_chat_template and getattr(tokenizer, "chat_template", None):
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if system_prompt:
            return f"System:\n{system_prompt}\n\nUser:\n{prompt}\n\nAssistant:\n"
        return f"User:\n{prompt}\n\nAssistant:\n"

    def _require_model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError("The Hugging Face model has not been loaded yet.")
        return self._model

    def _require_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("The tokenizer has not been loaded yet.")
        return self._tokenizer

    def _inspect_model_topology(self, model: PreTrainedModel) -> ModelTopology:
        layers: list[LayerTopology] = []
        seen_layers: set[int] = set()

        for module_name, module in model.named_modules():
            if not _is_attention_module(module_name, module):
                continue
            layer_index = _extract_layer_index(module_name)
            if layer_index is None or layer_index in seen_layers:
                continue

            num_heads = int(getattr(module, "num_heads", 0))
            head_dim = int(getattr(module, "head_dim", 0))
            if num_heads <= 0 or head_dim <= 0:
                continue

            seen_layers.add(layer_index)
            layers.append(
                LayerTopology(
                    layer_index=layer_index,
                    layer_name=f"Layer_{layer_index + 1}",
                    head_count=num_heads,
                    head_dim=head_dim,
                )
            )

        layers.sort(key=lambda item: item.layer_index)
        total_heads = sum(layer.head_count for layer in layers)

        return ModelTopology(
            model_name=self._settings.hf_model_name,
            device=self._device,
            total_layers=len(layers),
            total_heads=total_heads,
            layers=layers,
        )


class NeuralInferenceEngine:
    def __init__(self, settings: InferenceSettings | None = None) -> None:
        self.settings = settings or InferenceSettings()
        self._ollama = OllamaClient(self.settings)
        self._hooked_runner = HookedTransformerRunner(self.settings)

    async def startup(self) -> None:
        if self.settings.preload_shadow_model:
            await self._hooked_runner.ensure_loaded()

    async def shutdown(self) -> None:
        await self._ollama.close()
        await self._hooked_runner.close()

    async def get_model_topology(self) -> ModelTopology:
        return await self._hooked_runner.get_model_topology()

    async def set_analysis_model(self, model_name: str) -> None:
        await self._hooked_runner.use_model(model_name)

    async def is_ollama_available(self) -> bool:
        return await self._ollama.is_available()

    async def set_masked_heads(self, masked_heads: set[tuple[int, int]]) -> list[HeadMask]:
        return await self._hooked_runner.replace_masked_heads(masked_heads)

    def get_masked_heads(self) -> list[HeadMask]:
        return self._hooked_runner.get_masked_heads()

    async def stream(
        self,
        request: GenerationRequest,
        *,
        step_listener: StepListener | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        if request.trace_model_name:
            await self.set_analysis_model(request.trace_model_name)

        ollama_available = self.settings.use_ollama_if_available and await self._ollama.is_available()
        if request.execution_mode == TraceExecutionMode.FAITHFUL:
            await self._hooked_runner.ensure_loaded()

        tracer_ready = self._hooked_runner.is_ready()
        use_ollama = request.execution_mode != TraceExecutionMode.FAITHFUL and ollama_available

        if use_ollama:
            shadow_request = request.model_copy(
                update={
                    "max_new_tokens": self.settings.shadow_max_new_tokens or request.max_new_tokens,
                }
            )
            # Only start a shadow HF capture task if the tracer appears to be
            # loaded (tokenizer and model present). If the tracer is not
            # available (for example the configured HF name looks like an
            # Ollama identifier or was not preloaded), skip shadow capture and
            # return a proxy trace at the end.
            shadow_task = None
            if getattr(self._hooked_runner, "_tokenizer", None) is not None and getattr(
                self._hooked_runner, "_model", None
            ) is not None:
                shadow_task = asyncio.create_task(
                    self._hooked_runner.capture_trace(
                        shadow_request,
                        analysis_mode=AnalysisMode.SHADOW,
                        step_listener=step_listener,
                    )
                )
            else:
                LOGGER.warning(
                    "Shadow tracer appears to be unavailable; skipping HF capture for this run."
                )

            try:
                ollama_output_parts: list[str] = []
                async for chunk in self._ollama.stream_generate(request):
                    if chunk.token:
                        ollama_output_parts.append(chunk.token)
                    yield chunk

                if shadow_task is None:
                    # No shadow trace available — synthesize a proxy trace so
                    # the UI and lineage pipeline still receive a payload.
                    trace = AttentionTrace(
                        source_prompt=request.prompt,
                        generation_model=self.settings.ollama_model,
                        analysis_model=self.settings.hf_model_name,
                        generation_backend=GenerationBackend.OLLAMA,
                        analysis_mode=AnalysisMode.SHADOW,
                        trace_fidelity=TraceFidelity.PROXY,
                        prompt_token_count=1,
                        analysis_error="Shadow tracer unavailable or not preloaded.",
                    )
                    match_score = None
                    final_fidelity = TraceFidelity.PROXY
                    fidelity_reason = "No shadow HF trace was available for comparison."
                else:
                    try:
                        trace = await shadow_task
                    except Exception as exc:
                        LOGGER.exception("Shadow attention capture failed while Ollama generation succeeded.")
                        # Build a minimal trace payload first, then attach optional
                        # metadata so that any model/schema differences do not raise
                        # earlier pydantic validation errors.
                        trace = AttentionTrace(
                            source_prompt=request.prompt,
                            generation_model=self.settings.ollama_model,
                            analysis_model=self.settings.hf_model_name,
                            generation_backend=GenerationBackend.OLLAMA,
                            analysis_mode=AnalysisMode.SHADOW,
                            trace_fidelity=TraceFidelity.PROXY,
                            prompt_token_count=1,
                            analysis_error=str(exc),
                        )
                        try:
                            trace = trace.model_copy(
                                update={
                                    "match_score": None,
                                    "fidelity_reason": f"Shadow capture failed: {exc}",
                                }
                            )
                        except Exception:
                            # If model_copy fails due to schema mismatch, silently
                            # continue with the minimal trace so we still return
                            # a usable payload rather than raising further.
                            LOGGER.debug(
                                "Could not attach match metadata to trace; continuing with minimal trace."
                            )
                    # Compute a similarity score between the Ollama output and the
                    # shadow HF trace. If they match (within a high threshold) we
                    # can safely promote the trace to exact evidence; otherwise keep
                    # it labeled as proxy and include a match score.
                    try:
                        ollama_text = "".join(ollama_output_parts)
                        shadow_text = trace.generated_text or ""
                        norm_ollama = " ".join(ollama_text.split())
                        norm_shadow = " ".join(shadow_text.split())
                        match_score = float(difflib.SequenceMatcher(None, norm_ollama, norm_shadow).ratio())
                    except Exception:
                        LOGGER.exception("Failed to compute match score between Ollama and shadow HF trace.")
                        match_score = None

                    final_fidelity = TraceFidelity.PROXY
                    fidelity_reason: str | None = None
                    if match_score is not None:
                        if match_score >= 0.995:
                            final_fidelity = TraceFidelity.EXACT
                            fidelity_reason = f"Shadow HF trace matched Ollama output (score={match_score:.3f})."
                        else:
                            fidelity_reason = f"Ollama vs HF mismatch (score={match_score:.3f})."

                    summary = trace.summary
                    if summary is not None:
                        summary = summary.model_copy(
                            update={
                                "explanation": _build_trace_summary_explanation(
                                    final_fidelity,
                                    summary.dominant_heads,
                                    summary.influential_tokens,
                                    summary.masked_heads_applied,
                                )
                            }
                        )
                    trace = trace.model_copy(
                        update={
                            "generation_model": self.settings.ollama_model,
                            "analysis_model": self.settings.hf_model_name,
                            "generation_backend": GenerationBackend.OLLAMA,
                            "analysis_mode": AnalysisMode.SHADOW,
                            "trace_fidelity": final_fidelity,
                            "match_score": match_score,
                            "fidelity_reason": fidelity_reason,
                            "evidence_quality": _assess_trace_evidence(
                                trace,
                                trace_fidelity=final_fidelity,
                                match_score=match_score,
                                analysis_error=trace.analysis_error,
                            ),
                            "summary": summary,
                        }
                    )
                if trace.evidence_quality is None:
                    trace = trace.model_copy(
                        update={
                            "evidence_quality": _assess_trace_evidence(
                                trace,
                                trace_fidelity=final_fidelity,
                                match_score=match_score,
                                analysis_error=trace.analysis_error,
                            ),
                            "fidelity_reason": fidelity_reason,
                        }
                    )

                yield GenerationChunk(
                    backend=GenerationBackend.OLLAMA,
                    model=self.settings.ollama_model,
                    done=True,
                    trace=trace,
                )
                return
            except Exception:
                if shadow_task is not None:
                    shadow_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await shadow_task
                raise

        if not tracer_ready:
            await self._hooked_runner.ensure_loaded()
            tracer_ready = self._hooked_runner.is_ready()

        if not tracer_ready:
            raise RuntimeError(
                "Exact black-box tracing requires a loaded Hugging Face tokenizer/model. "
                f"The configured SYNAPSE_HF_MODEL_NAME='{self.settings.hf_model_name}' could not be loaded. "
                "Use a real Hugging Face repo id for SYNAPSE_HF_MODEL_NAME, or enable Ollama and use "
                "execution_mode='auto' for proxy-only generation."
            )

        async for chunk in self._hooked_runner.generate_stream(
            request,
            analysis_mode=AnalysisMode.INLINE,
            step_listener=step_listener,
        ):
            yield chunk

    async def generate(
        self,
        request: GenerationRequest,
        *,
        step_listener: StepListener | None = None,
    ) -> InferenceResponse:
        output_chunks: list[str] = []
        final_chunk: GenerationChunk | None = None

        async for chunk in self.stream(request, step_listener=step_listener):
            if chunk.token:
                output_chunks.append(chunk.token)
            if chunk.done:
                final_chunk = chunk

        if final_chunk is None or final_chunk.trace is None:
            raise RuntimeError("The inference engine finished without emitting a final trace payload.")

        return InferenceResponse(
            backend=final_chunk.backend,
            generation_model=final_chunk.trace.generation_model,
            analysis_model=final_chunk.trace.analysis_model,
            text="".join(output_chunks),
            trace=final_chunk.trace,
        )


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested_dtype: str | None, device: str) -> torch.dtype:
    if requested_dtype:
        normalized = requested_dtype.lower()
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        try:
            return mapping[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported SYNAPSE_HF_DTYPE value: {requested_dtype}") from exc

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device == "mps":
        return torch.float16

    return torch.float32


def _is_attention_module(module_name: str, module: nn.Module) -> bool:
    lowered_name = module_name.lower()
    lowered_type = module.__class__.__name__.lower()
    if not any(token in lowered_name for token in ("self_attn", ".attn", "attention")):
        return False
    return hasattr(module, "num_heads") and ("attn" in lowered_type or "attention" in lowered_type)


def _extract_layer_index(module_name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(module_name)
        if match:
            return int(match.group(1))
    return None


def _extract_attention_tensor(output: Any) -> Tensor | None:
    if isinstance(output, Tensor) and output.ndim == 4:
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, Tensor) and item.ndim == 4:
                return item

    for attribute_name in ("attn_weights", "attention_probs", "attention_weights"):
        attribute = getattr(output, attribute_name, None)
        if isinstance(attribute, Tensor) and attribute.ndim == 4:
            return attribute

    return None


def _matches_any_stop_sequence(text: str, stop_sequences: list[str]) -> bool:
    if not text or not stop_sequences:
        return False
    return any(stop_sequence and text.endswith(stop_sequence) for stop_sequence in stop_sequences)


def _group_mask_heads(masked_heads: set[tuple[int, int]]) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = {}
    for layer_index, head_index in masked_heads:
        grouped.setdefault(layer_index, set()).add(head_index)
    return grouped


def _format_mask_names(masked_heads_by_layer: dict[int, set[int]]) -> list[str]:
    formatted: list[str] = []
    for layer_index, head_indices in sorted(masked_heads_by_layer.items()):
        for head_index in sorted(head_indices):
            formatted.append(f"Layer_{layer_index + 1}:Head_{head_index + 1}")
    return formatted


def _format_token_window(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
    context_window: int,
) -> list[str]:
    window_ids = token_ids[-context_window:]
    raw_tokens = tokenizer.convert_ids_to_tokens(window_ids, skip_special_tokens=False)
    return [_normalize_display_token(token) for token in raw_tokens]


def _normalize_display_token(token: str) -> str:
    cleaned = (
        token.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("Ċ", "\\n")
        .replace("ĉ", "\\t")
    )
    if cleaned == "":
        return "<empty>"
    if cleaned.isspace():
        return "<space>"
    return cleaned


def _dedupe_tokens(tokens: Any) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw_token in tokens:
        token = str(raw_token)
        if not _is_informative_token(token) or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _is_informative_token(token: str) -> bool:
    normalized = token.strip()
    if not normalized:
        return False
    if normalized in {"<space>", "<empty>", "\\n", "\\t"}:
        return False
    return True


def _quote_token(token: str) -> str:
    compact = token.replace("\n", "\\n")
    return f'"{compact}"'


def _build_step_explanation(
    *,
    generated_token: str,
    ranked_layers: list[LayerActivation],
    evidence_tokens: list[str],
    masked_head_names: list[str],
) -> str | None:
    if not ranked_layers:
        return None

    top_routes: list[str] = []
    for layer in ranked_layers[:2]:
        if not layer.top_heads:
            continue
        head = layer.top_heads[0]
        route = f"{layer.layer_name}/{head.head_name}"
        if head.top_source_tokens:
            route += " attending to " + ", ".join(_quote_token(token) for token in head.top_source_tokens[:2])
        top_routes.append(route)

    if not top_routes:
        return None

    generated_label = _quote_token(generated_token) if generated_token else "the next token"
    explanation_parts = [
        f"{generated_label} was driven most strongly by " + " and ".join(top_routes) + "."
    ]
    if evidence_tokens:
        explanation_parts.append(
            "Key evidence tokens were "
            + ", ".join(_quote_token(token) for token in evidence_tokens[:4])
            + "."
        )
    if masked_head_names:
        explanation_parts.append(f"{len(masked_head_names)} masked heads constrained this step.")
    return " ".join(explanation_parts)


def _build_trace_summary_explanation(
    fidelity: TraceFidelity,
    dominant_heads: list[str],
    influential_tokens: list[str],
    masked_heads_applied: list[str],
) -> str:
    explanation_parts = [
        "This run is grounded in exact inline tracing."
        if fidelity == TraceFidelity.EXACT
        else "This run uses proxy tracing from the shadow model, so treat it as strong evidence rather than exact causality."
    ]
    if dominant_heads:
        explanation_parts.append(
            "The dominant neural route repeatedly passed through "
            + ", ".join(dominant_heads[:3])
            + "."
        )
    if influential_tokens:
        explanation_parts.append(
            "The model concentrated most on "
            + ", ".join(_quote_token(token) for token in influential_tokens[:5])
            + "."
        )
    if masked_heads_applied:
        explanation_parts.append(f"{len(masked_heads_applied)} masked heads were enforced during the run.")
    return " ".join(explanation_parts)


def _assess_trace_evidence(
    trace: AttentionTrace,
    *,
    trace_fidelity: TraceFidelity | None = None,
    match_score: float | None = None,
    analysis_error: str | None = None,
) -> EvidenceQuality:
    fidelity = trace_fidelity or trace.trace_fidelity
    resolved_match_score = trace.match_score if match_score is None else match_score
    resolved_error = analysis_error or trace.analysis_error
    has_steps = bool(trace.steps)
    has_lineage = any(step.high_activation_path for step in trace.steps)

    score = 0.2
    if has_steps:
        score += 0.25
    if has_lineage:
        score += 0.15
    if trace.summary is not None:
        score += 0.1
    if fidelity == TraceFidelity.EXACT:
        score += 0.25
    elif resolved_match_score is not None:
        score += max(0.0, min(resolved_match_score, 1.0)) * 0.15
    if resolved_error:
        score -= 0.2
    score = max(0.0, min(score, 1.0))

    if score >= 0.8:
        label = "high"
    elif score >= 0.55:
        label = "medium"
    else:
        label = "low"

    if fidelity == TraceFidelity.EXACT:
        exactness = "Generated tokens and captured activations came from the same hooked model run."
    else:
        exactness = (
            "Generation and tracing were decoupled; the trace should be treated as proxy evidence "
            "unless the shadow output closely matches the live output."
        )

    gaps: list[str] = []
    actions: list[str] = []
    if fidelity == TraceFidelity.PROXY:
        gaps.append("Proxy trace cannot prove that the displayed heads caused the live output.")
        actions.append("Use execution_mode='faithful' for exact hooked generation when latency permits.")
    if resolved_match_score is None and trace.generation_backend == GenerationBackend.OLLAMA:
        gaps.append("No generator-vs-shadow match score is available for this run.")
        actions.append("Preload a compatible Hugging Face shadow model before using fast Ollama mode.")
    elif resolved_match_score is not None and resolved_match_score < 0.995:
        gaps.append(f"Shadow output diverged from live output (match_score={resolved_match_score:.3f}).")
        actions.append("Align the Ollama and Hugging Face model families or lower claims to proxy telemetry.")
    if not has_steps:
        gaps.append("No per-token attention steps were captured.")
        actions.append("Verify Hugging Face tracing is loaded and output_attentions is supported.")
    if not has_lineage:
        gaps.append("No high-activation lineage path was emitted.")
        actions.append("Check capture thresholds and attention hook compatibility for the selected model.")
    gaps.append("Attention weights are mechanism telemetry, not complete causal proof by themselves.")
    actions.append("Validate suspicious heads with ablation or counterfactual replay before governance decisions.")

    return EvidenceQuality(
        score=round(score, 3),
        label=label,
        exactness=exactness,
        causal_validation=(
            "validated_by_same_run_hooks"
            if fidelity == TraceFidelity.EXACT and has_lineage
            else "requires_ablation_or_replay"
        ),
        black_box_gaps=_dedupe_tokens(gaps),
        recommended_next_actions=_dedupe_tokens(actions),
    )
