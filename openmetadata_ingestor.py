"""Simple metadata ingestor wrappers that post to ingestion endpoints.

Assumes a metadata ingestor service accepts POST /ingest/inference and POST /ingest/explanation
Payload shapes are application-specific; this wrapper forwards the payload with an Authorization header if OPENMETADATA_API_KEY is set.
"""
import os
import requests

METADATA_INGESTOR_URL = os.environ.get("METADATA_INGESTOR_URL")
OPENMETADATA_API_KEY = os.environ.get("OPENMETADATA_API_KEY")


def post_inference(payload: dict) -> dict:
    if not METADATA_INGESTOR_URL:
        print("[ingestor] METADATA_INGESTOR_URL not configured; skipping ingestion")
        return {}
    url = METADATA_INGESTOR_URL.rstrip("/") + "/ingest/inference"
    headers = {"Content-Type": "application/json"}
    if OPENMETADATA_API_KEY:
        headers["Authorization"] = f"Bearer {OPENMETADATA_API_KEY}"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except Exception as e:
        print(f"[ingestor] failed to post inference: {e}")
        return {}


def post_explanation(payload: dict) -> dict:
    if not METADATA_INGESTOR_URL:
        print("[ingestor] METADATA_INGESTOR_URL not configured; skipping ingestion")
        return {}
    url = METADATA_INGESTOR_URL.rstrip("/") + "/ingest/explanation"
    headers = {"Content-Type": "application/json"}
    if OPENMETADATA_API_KEY:
        headers["Authorization"] = f"Bearer {OPENMETADATA_API_KEY}"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except Exception as e:
        print(f"[ingestor] failed to post explanation: {e}")
        return {}
