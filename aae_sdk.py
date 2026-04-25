"""AAE SDK - lightweight inference recorder

Usage:
- Configure via env vars: ARTIFACT_BASE_URI, METADATA_INGESTOR_URL, OPENMETADATA_API_KEY
- Recorder uploads artifacts to local ./artifacts and returns URIs prefixed by ARTIFACT_BASE_URI
- Posts metadata to metadata ingestor endpoints when available
"""
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import requests

ARTIFACT_DIR = Path.cwd() / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


class InferenceRecorder:
    def __init__(self, ingest_url: str = None, artifact_base: str = None):
        # read env vars at init time to allow dynamic overrides
        self.ingest_url = ingest_url or os.environ.get("METADATA_INGESTOR_URL")
        self.artifact_base = artifact_base or os.environ.get("ARTIFACT_BASE_URI", "http://localhost:9000")
        self.openmetadata_api_key = os.environ.get("OPENMETADATA_API_KEY")

    def _write_artifact(self, relative_path: str, data) -> str:
        path = ARTIFACT_DIR / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # normalize artifact base + relative path
        return f"{self.artifact_base.rstrip('/')}/{relative_path}"

    def post(self, endpoint: str, payload: dict):
        if not self.ingest_url:
            print("[aae_sdk] METADATA_INGESTOR_URL not set; skipping POST to metadata ingestor")
            return None
        url = self.ingest_url.rstrip("/") + endpoint
        headers = {"Content-Type": "application/json"}
        if self.openmetadata_api_key:
            headers["Authorization"] = f"Bearer {self.openmetadata_api_key}"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except requests.RequestException as e:
            print(f"[aae_sdk] failed to POST to {url}: {e}")
            return None

    def record_inference(self, model_id: str, input_text: str, model_output: dict, explanation: dict) -> dict:
        inference_id = f"inf-{uuid.uuid4()}"
        timestamp = datetime.utcnow().isoformat() + "Z"

        input_payload = {"inference_id": inference_id, "timestamp": timestamp, "input_text": input_text}
        input_rel = f"inputs/{inference_id}.json"
        input_uri = self._write_artifact(input_rel, input_payload)

        explanation_rel = f"explain/{inference_id}.json"
        explanation_payload = {"inference_id": inference_id, "explanation": explanation}
        explanation_uri = self._write_artifact(explanation_rel, explanation_payload)

        inference_metadata = {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "model_id": model_id,
            "input_uri": input_uri,
            "response": model_output.get("answer"),
            "confidence": model_output.get("confidence"),
            "explanation_uri": explanation_uri,
        }

        # POST metadata to metadata ingestor if configured
        self.post("/ingest/inference", inference_metadata)

        explanation_metadata = {
            "explanation_id": f"exp-{uuid.uuid4()}",
            "inference_id": inference_id,
            "method": explanation.get("method"),
            "summary": explanation.get("summary"),
            "artifact_uri": explanation_uri,
        }
        self.post("/ingest/explanation", explanation_metadata)

        return {"inference_id": inference_id, "input_uri": input_uri, "explanation_uri": explanation_uri}
