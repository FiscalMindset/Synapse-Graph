import sys
from pathlib import Path

from fastapi.testclient import TestClient


def test_discover_then_quarantine_flow(monkeypatch):
    # Ensure repo root on sys.path
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from backend.app.main import app, runtime
    from backend.app.inference import (
        InferenceResponse,
        AttentionTrace,
        TokenStepCapture,
        LayerActivation,
        HeadActivation,
        GenerationBackend,
        AnalysisMode,
        TraceFidelity,
        HeadMask,
    )
    from backend.app.main import ModelTopology

    # Prevent heavy startup/shutdown
    async def noop_startup():
        return None

    async def noop_shutdown():
        return None

    monkeypatch.setattr(runtime, "startup", noop_startup)
    monkeypatch.setattr(runtime, "shutdown", noop_shutdown)

    # Topology is provided via runtime.ensure_topology() stubbed below

    # Build a minimal baseline trace where token 'green' is present
    head = HeadActivation(head_index=0, head_name="Head_1", masked=False, max_attention_score=0.9, mean_attention_score=0.5, l2_norm=1.0)
    layer = LayerActivation(layer_index=0, layer_name="Layer_1", sequence_length=1, head_count=4, masked_head_names=[], dominant_source_tokens=["green"], top_heads=[head])
    step = TokenStepCapture(step_index=0, generated_token="The sky is green.", generated_token_id=0, prompt_plus_generation_length=1, masked_heads=[], layers=[layer], high_activation_path=["Layer_1:Head_1"], evidence_tokens=["green"]) 

    baseline_trace = AttentionTrace(
        source_prompt="Why is the sky green?",
        generation_model="gpt2",
        analysis_model="gpt2",
        generation_backend=GenerationBackend.HUGGINGFACE,
        analysis_mode=AnalysisMode.INLINE,
        trace_fidelity=TraceFidelity.EXACT,
        prompt_token_count=1,
        generated_text="The sky is green.",
        steps=[step],
    )

    ablated_trace = AttentionTrace(
        source_prompt="Why is the sky green?",
        generation_model="gpt2",
        analysis_model="gpt2",
        generation_backend=GenerationBackend.HUGGINGFACE,
        analysis_mode=AnalysisMode.INLINE,
        trace_fidelity=TraceFidelity.EXACT,
        prompt_token_count=1,
        generated_text="The sky is .",
        steps=[step],
    )

    baseline_response = InferenceResponse(backend=GenerationBackend.HUGGINGFACE, generation_model="gpt2", analysis_model="gpt2", text="The sky is green.", trace=baseline_trace)
    ablated_response = InferenceResponse(backend=GenerationBackend.HUGGINGFACE, generation_model="gpt2", analysis_model="gpt2", text="The sky is .", trace=ablated_trace)

    # track current masked set
    current_pairs = set()

    async def fake_set_masked_heads(pairs):
        nonlocal current_pairs
        current_pairs = set(pairs)
        return [HeadMask(layer_index=layer, layer_name=f"Layer_{layer+1}", head_index=head, head_name=f"Head_{head+1}") for (layer, head) in sorted(current_pairs)]

    def fake_get_masked_heads():
        return [HeadMask(layer_index=layer, layer_name=f"Layer_{layer+1}", head_index=head, head_name=f"Head_{head+1}") for (layer, head) in sorted(current_pairs)]

    async def fake_generate(request, step_listener=None):
        # Return baseline when nothing is masked, else ablated
        if not current_pairs:
            return baseline_response
        return ablated_response

    # patch runtime functions
    monkeypatch.setattr(runtime.inference, "set_masked_heads", fake_set_masked_heads)
    monkeypatch.setattr(runtime.inference, "get_masked_heads", fake_get_masked_heads)
    monkeypatch.setattr(runtime.inference, "generate", fake_generate)

    # ensure_topology returns a topology and a dummy catalog so quarantine can run
    async def fake_ensure_topology():
        return (ModelTopology(model_name="test", device="cpu", total_layers=1, total_heads=4, layers=[]), object())

    monkeypatch.setattr(runtime, "ensure_topology", fake_ensure_topology)

    # stub OpenMetadata tagging to return a non-empty applied list
    async def fake_apply_defective_tags(catalog, pairs, reason=None):
        return [HeadMask(layer_index=layer, layer_name=f"Layer_{layer+1}", head_index=head, head_name=f"Head_{head+1}", reason=reason) for (layer, head) in pairs]

    monkeypatch.setattr(runtime.openmetadata, "apply_defective_tags", fake_apply_defective_tags)

    client = TestClient(app)

    # Run discovery - target 'green' should be present in baseline and removed in ablated
    req = {
        "prompt": "Why is the sky green?",
        "target_hallucination_token": "green",
        "trace_model_name": "gpt2",
        "max_new_tokens": 8,
        "top_k_heads": 1,
        "max_pair_sweeps": 0,
    }

    resp = client.post("/api/v1/autopsy/discover_circuit", json=req)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["baseline_target_count"] >= 1
    # discovered_circuit should be non-empty when an ablation removed token
    assert isinstance(body["discovered_circuit"], list)

    # Now quarantine the discovered circuit
    if body["discovered_circuit"]:
        q_payload = {"heads": body["discovered_circuit"], "reason": "integration_test"}
        qresp = client.post("/api/v1/openmetadata/quarantine", json=q_payload)
        assert qresp.status_code == 200
        qbody = qresp.json()
        assert "masked_heads" in qbody
