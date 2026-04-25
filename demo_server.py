"""Minimal demo server exposing /query and demonstrating the AAE flow.

Run: uvicorn demo_server:app --host 0.0.0.0 --port 8000

Environment variables used:
- ARTIFACT_BASE_URI (optional)
- METADATA_INGESTOR_URL (optional)
- OPENMETADATA_API_KEY (optional)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from aae_sdk import InferenceRecorder
import re

app = FastAPI(title="AAE Demo Server")
recorder = InferenceRecorder()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
async def health():
    return {"status": "ok"}


def simple_answer_and_explain(query: str) -> (Dict[str, Any], Dict[str, Any]):
    q = query.lower()
    # very small knowledge base
    if "einstein" in q:
        answer = "Albert Einstein co-invented the Einstein–Szilard refrigerator and is known for the theory of relativity."
        confidence = 0.92
        sources = [
            {"type": "document", "name": "wikipedia:Albert_Einstein", "link": "https://en.wikipedia.org/wiki/Albert_Einstein", "confidence": 0.95}
        ]
    else:
        answer = "I don't know. This demo only answers Einstein questions."
        confidence = 0.35
        sources = []

    # simple token attribution explanation: score tokens that match keywords
    tokens = re.findall(r"\w+", query)
    keywords = {"einstein", "refrigerator", "szilard", "invention", "relativity"}
    attributions = []
    for t in tokens:
        score = 1.0 if t.lower() in keywords else 0.05
        attributions.append({"token": t, "score": score})

    # pseudo activations and attention (deterministic per-query)
    import random
    seed = sum(ord(c) for c in query) % (2**32)
    rng = random.Random(seed)
    activations = {
        "layer_1": [round(rng.random(), 4) for _ in range(8)],
        "layer_2": [round(rng.random(), 4) for _ in range(6)]
    }
    # attention matrix between tokens (nxn)
    n = len(tokens)
    attention = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(round(rng.random(), 3))
        # normalize row
        row_sum = sum(row) or 1.0
        row = [round(x / row_sum, 3) for x in row]
        attention.append(row)

    explanation = {
        "method": "token-attribution+activations",
        "summary": "Top tokens: Einstein, refrigerator, Szilard",
        "attributions": attributions,
        "activations": activations,
        "attention": attention
    }

    # add a small counterfactual example for demo
    counterfactual = {"changed_input": "what invention done by szilard", "changed_prediction": "Szilard co-invented the refrigerator with Einstein", "delta_confidence": -0.02}
    explanation["counterfactual"] = counterfactual

    model_output = {"answer": answer, "confidence": confidence, "sources": sources}
    return model_output, explanation


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    model_id = "demo-wiki-qa-v1"
    model_output, explanation = simple_answer_and_explain(req.query)

    # record inference + explanation artifacts and (optionally) post metadata
    record = recorder.record_inference(model_id=model_id, input_text=req.query, model_output=model_output, explanation=explanation)

    response = {
        "answer": model_output["answer"],
        "model_version": model_id,
        "confidence": model_output["confidence"],
        "sources": model_output.get("sources", []),
        "provenance": [{
            "inference_id": record.get("inference_id"),
            "artifact_link": record.get("explanation_uri")
        }],
        "explanations": [{
            "id": record.get("explanation_uri"),
            "method": explanation.get("method"),
            "summary": explanation.get("summary"),
            "artifact": record.get("explanation_uri")
        }]
    }
    return response
