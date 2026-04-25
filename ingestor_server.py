"""Simple metadata ingestor service for demo.

Endpoints:
- POST /ingest/inference  -> stores JSON to artifacts/ingested/inference/{id}.json
- POST /ingest/explanation -> stores JSON to artifacts/ingested/explanation/{id}.json
- GET /ingested -> lists stored ingested files

Optionally forwards to OpenMetadata if OPENMETADATA_HOST and OPENMETADATA_API_KEY are set, but by default only persists locally.
"""
from fastapi import FastAPI, Request, HTTPException
from pathlib import Path
import json
import os
from typing import Any

app = FastAPI(title="AAE Metadata Ingestor")
BASE = Path.cwd() / "artifacts" / "ingested"
(BASE / "inference").mkdir(parents=True, exist_ok=True)
(BASE / "explanation").mkdir(parents=True, exist_ok=True)

OPENMETADATA_HOST = os.environ.get("OPENMETADATA_HOST")
OPENMETADATA_API_KEY = os.environ.get("OPENMETADATA_API_KEY")


@app.post("/ingest/inference")
async def ingest_inference(req: Request):
    payload = await req.json()
    inference_id = payload.get("inference_id") or payload.get("id")
    if not inference_id:
        raise HTTPException(status_code=400, detail="inference_id required")
    path = BASE / "inference" / f"{inference_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # optional: forward to OpenMetadata (not implemented automatic mapping)
    return {"stored": str(path)}


@app.post("/ingest/explanation")
async def ingest_explanation(req: Request):
    payload = await req.json()
    explanation_id = payload.get("explanation_id") or payload.get("id")
    if not explanation_id:
        raise HTTPException(status_code=400, detail="explanation_id required")
    path = BASE / "explanation" / f"{explanation_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return {"stored": str(path)}


@app.get("/ingested")
async def list_ingested():
    items = {"inference": [], "explanation": []}
    for p in (BASE / "inference").glob("*.json"):
        items["inference"].append(p.name)
    for p in (BASE / "explanation").glob("*.json"):
        items["explanation"].append(p.name)
    return items
