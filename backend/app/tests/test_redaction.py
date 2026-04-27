import re
import json
import base64

from backend.app.om_client import _build_synthetic_sql, redact_for_storage
from backend.app.inference import TokenStepCapture


def test_redact_embedded_meta():
    # Build a fake step that contains an email in evidence tokens and explanation
    step = TokenStepCapture(
        step_index=1,
        generated_token="hello",
        generated_token_id=1,
        prompt_plus_generation_length=10,
        masked_heads=[],
        layers=[],
        high_activation_path=[],
        evidence_tokens=["alice@company.com", "normal"],
        evidence_positions=[0],
        evidence_token_ids=[0],
        evidence_token_attention={0: 0.9},
        explanation="Contact alice@company.com for details",
    )

    sql = _build_synthetic_sql("sess-1", "Hello alice@company.com", step, "Layer_1", session_meta={"prompt": "Hello alice@company.com", "_synapse_session_meta": {"generation_seed": 42}})

    # Extract base64 blob
    m = re.search(r"SYNAPSE_META_B64:\s*([A-Za-z0-9+/=\n\r\s]+)\s*\*/", sql)
    assert m, "SYNAPSE_META_B64 not found in SQL"
    b64 = m.group(1).strip()
    decoded = base64.b64decode(b64).decode("utf-8")
    meta = json.loads(decoded)

    # Ensure email was redacted in the embedded metadata
    assert "<REDACTED_EMAIL>" in json.dumps(meta) or "REDACTED" in json.dumps(meta)


def test_redact_for_storage_recursive():
    payload = {
        "prompt": "send to bob@example.org and 4111 1111 1111 1111",
        "nested": {"contact": "alice@company.com", "long": "x" * 1000},
    }
    redacted = redact_for_storage(payload)
    s = json.dumps(redacted)
    assert "<REDACTED_EMAIL>" in s
    assert "<REDACTED_NUMBER>" in s or "<TRUNCATED>" in s
