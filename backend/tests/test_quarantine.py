import json
import sys
from pathlib import Path

import pytest

from fastapi.testclient import TestClient


def test_quarantine_endpoint(monkeypatch):
    # Ensure repo root is on sys.path so `backend` package imports work
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    # Import app and runtime from the running module
    from backend.app.main import app, runtime, HeadMask, ModelTopology

    # Prevent heavy startup/shutdown during test by stubbing lifecycle methods
    async def noop_startup():
        return None

    async def noop_shutdown():
        return None

    monkeypatch.setattr(runtime, "startup", noop_startup)
    monkeypatch.setattr(runtime, "shutdown", noop_shutdown)

    # Ensure ensure_topology returns a simple topology and a dummy catalog object
    async def fake_ensure_topology():
        return (
            ModelTopology(model_name="test", device="cpu", total_layers=0, total_heads=0, layers=[]),
            object(),
        )

    monkeypatch.setattr(runtime, "ensure_topology", fake_ensure_topology)

    # Stub openmetadata.apply_defective_tags to echo back HeadMask objects
    async def fake_apply_defective_tags(catalog, pairs, reason=None):
        return [HeadMask(layer_index=layer, layer_name=f"Layer_{layer+1}", head_index=head, head_name=f"Head_{head+1}", reason=reason) for (layer, head) in pairs]

    monkeypatch.setattr(runtime.openmetadata, "apply_defective_tags", fake_apply_defective_tags)

    # Stub inference.get_masked_heads and set_masked_heads
    monkeypatch.setattr(runtime.inference, "get_masked_heads", lambda: [])

    async def fake_set_masked_heads(pairs):
        return [HeadMask(layer_index=layer, layer_name=f"Layer_{layer+1}", head_index=head, head_name=f"Head_{head+1}") for (layer, head) in sorted(pairs)]

    monkeypatch.setattr(runtime.inference, "set_masked_heads", fake_set_masked_heads)

    client = TestClient(app)

    payload = {"heads": [{"layer_index": 0, "layer_name": "Layer_1", "head_index": 0, "head_name": "Head_1", "activation_score": 0.1}], "reason": "unit-test"}

    response = client.post("/api/v1/openmetadata/quarantine", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] in (True, False)
    assert "masked_heads" in body
