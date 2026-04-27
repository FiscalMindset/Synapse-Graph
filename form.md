# Synapse-Graph — Hackathon Submission Form

---

## Project Description

Synapse-Graph transforms a running transformer model into an observable, governable system by:

- **Capturing per-token, per-layer, per-head attention activations** via hook-instrumented PyTorch "shadow" tracer using `register_forward_hook`
- **Translating activations into OpenMetadata entities** (model → database → schema → layer tables → head columns) and ingesting lineage edges for active neural routes
- **Running causal ablation sweeps** (single-head + pair-head) to surface heads that materially change outputs for a target hallucination token
- **Closing the governance loop**: operators tag defective heads in OpenMetadata (DEFECTIVE/QUARANTINED), and the backend converts those tags to runtime head masks applied to subsequent generations

**What makes it different**: Repurposes OpenMetadata as a governance plane for transformer internals. Makes AI interpretability an operational control plane, not just static visualization.

---

## GitHub link to project

https://github.com/FiscalMindset/Synapse-Graph

---

## Deployed link to project

- **GitHub Pages (HTML Presentation)**: https://fiscalmindset.github.io/Synapse-Graph/
- **Frontend**: `http://localhost:3000` (run locally)
- **Backend**: `http://localhost:8000` (run locally)

---

## YouTube video demo link

- **Complete Demo Walkthrough**: https://youtu.be/idOJYh6TUC8
- **Product Demo (Live Run)**: https://youtu.be/b78Y7RwvYeU

---

## About the project

### Problem Addressed

LLMs are powerful but operationally opaque. Current observability tools stop at prompts, tokens, latency, and logs. They cannot answer:

- Which layers and heads were most active for this response?
- Can we trace a "thought path" through the network?
- Can governance tools intervene on specific neural components?

### What We Built

A full-stack system consisting of:

1. **FastAPI Neural Proxy** — Orchestrates generation (Ollama preferred, HF fallback), hook-based tracing, and OpenMetadata sync
2. **PyTorch Shadow Tracer** — Uses `register_forward_hook` to capture per-layer, per-head attention activations in real time
3. **Next.js Dashboard** — Visualizes active neural path, streams outputs, displays masked heads, shows OpenMetadata sync state
4. **OpenMetadata Integration** — Maps transformer internals to metadata entities (model → database → schema → tables → columns)

### Impact

- Teams can answer "which heads produced this hallucination?"
- Operationalize mitigation at the head level rather than blocking the whole prompt
- Makes AI interpretability into an operational control plane with familiar data-platform primitives

---

## Tech Stack and Architecture

### Backend Tech Stack

```python
# Exact from backend/pyproject.toml
requires-python = ">=3.11,<3.13"

dependencies = [
    "fastapi>=0.115.0",          # HTTP API
    "torch>=2.4.0",               # Tracing engine
    "transformers>=4.46.0",        # Model interface
    "openmetadata-ingestion>=1.12.0", # Metadata SDK
    "httpx>=0.28.0",              # REST client
    "pydantic-settings>=2.7.0",    # Configuration
    "uvicorn[standard]>=0.32.0",    # ASGI server
    "accelerate>=1.1.0",           # HF acceleration
    "cachetools>=5.3.0",          # Caching
]
```

### Frontend Tech Stack

```json
{
  "dependencies": {
    "next": "^15.2.0",            // Framework
    "react": "^19.0.0",            // UI
    "@xyflow/react": "^12.4.4",    // Graph visualization
    "recharts": "^2.15.0",         // Charts
    "lucide-react": "^0.468.0"      // Icons
  },
  "devDependencies": {
    "tailwindcss": "^3.4.16",       // Styling
    "typescript": "^5.7.2"          // Type safety
  }
}
```

### Architecture (Runtime Flow)

```
┌──────────────────────────────────────────────────────┐
│              Operator Dashboard (Next.js)              │
└─────────────────────┬────────────────────────────────┘
                      │ REST + SSE
┌─────────────────────▼────────────────────────────────┐
│              Neural Proxy (FastAPI)                    │
│    Orchestrates generation + tracing + governance     │
└──────┬────────────────┬────────────────┬────────────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────┐  ┌────────────────┐  ┌────────────────┐
│   Ollama     │  │  HF Tracer     │  │  OpenMetadata  │
│ (Preferred) │  │ (PyTorch)     │  │  (Governance)  │
│             │  │               │  │                │
│ Generation │  │ Attention    │  │ Topology      │
│             │  │ activation    │  │ Lineage       │
│             │  │ capture      │  │ Tags→Masks   │
└─────────────┘  └────────────────┘  └────────────────┘
```

### Backend Files

- `backend/app/main.py` — FastAPI app with 13 REST/SSE endpoints
- `backend/app/inference.py` — Generation engine + PyTorch hook-based tracer
- `backend/app/om_client.py` — OpenMetadata client for topology and lineage

### Frontend Files

- `frontend/components/synapse-dashboard.tsx` — Main dashboard with discovery panel
- `frontend/components/synapse-graph.tsx` — @xyflow/react graph visualization
- `frontend/components/activation-chart.tsx` — Per-layer, per-head activation charts

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/generate/stream` | POST | SSE generation with trace steps |
| `/api/v1/autopsy/discover_circuit` | POST | Circuit discovery |
| `/api/v1/openmetadata/bootstrap` | POST | Bootstrap OpenMetadata catalog |
| `/api/v1/openmetadata/sync-defects` | POST | Sync tags to masks |
| `/api/v1/governance/local-mask` | POST | Set head mask |

---

## Demo Steps (for 2-3 minute video)

### Setup Commands

```bash
# Terminal 1: Backend
cd Synapse-Graph
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ./backend
cp backend/.env.example backend/.env
cd backend && python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm install && npm run dev
# Open: http://localhost:3000
```

### Demo Actions (60-90 seconds each)

1. **Start**: Boot dashboard, verify backend reports "Ollama live" or "HF fallback"
2. **Trace**: Submit prompt → watch synapse graph light up with active edges
3. **Discover**: Enter hallucination token → run circuit discovery
4. **Quarantine**: Click "Quarantine" on discovered head → push DEFECTIVE tags to OpenMetadata
5. **Verify**: Re-run prompt → show masked heads count increase, changed output

### What Judges Should Notice

- Neural internals mapped to OpenMetadata (tables = layers, columns = heads)
- Concrete causal signal: ablation removes/alters hallucinated token
- Simple governance loop: metadata tag → runtime mask → observable behavior change

---

## Learning and Growth

### Engineering Learnings

- **Hook-based tracing**: Balancing trace fidelity vs runtime overhead. Using `register_forward_hook` for non-invasive attention capture while maintaining generation speed.
- **Two-level masking**: Implemented attention tensor masking + projection masking for effective head ablation.
- **Snapshot semantics**: Designing safe concurrency for `AttentionTrace` snapshots across SSE streams.

### Product Learnings

- **Interpretability as governance**: Framing mechanistic interpretability as governance made it easier to explain value to non-research stakeholders.
- **Familiar primitives**: Using OpenMetadata concepts (schemas, tables, columns, lineage) lowered the barrier for understanding neural behavior.

### Open Challenges (Honest Assessment)

- Causality proof (correlation vs causation in ablation)
- Ground-truth validation datasets for hallucination circuits
- Scalability for larger models (O(n²) pair ablation)

---

## OpenMetadata Usage

### Topology Bootstrap

Creates synthetic OpenMetadata entities:
- **Service**: `Synapse_Neural_Service` (type: MySQL)
- **Database**: `{Model_Name}` (e.g., `Qwen_Qwen2.5-1.5B-Instruct`)
- **Schema**: `Transformer_Graph`
- **Tables**: `Prompt_Ingress`, `Response_Egress`, `Layer_N` (one per layer)
- **Columns**: `Head_N` (one per head, FLOAT type)

### Classification & Tags

- **Classification**: `SynapseQuarantine`
- **Tag**: `DEFECTIVE` (color: #39FF14)

### API Calls Made

```python
# Column-level tagging
POST /v1/tables/name/{fqn}/columns/{column}/tags

# Table-level fallback
POST /v1/tables/name/{fqn}/tags

# Fetch with tags
GET /v1/tables/name/{fqn}?fields=columns,tags

# Lineage ingestion
metadata.add_lineage()
```

### Lineage Ingestion

- Creates edges: `Prompt_Ingress` → `Layer_1` → ... → `Layer_N` → `Response_Egress`
- Stores activation path in SQL query field
- Only top 2 active heads per layer included in column lineage

### Governance Flow

```
1. Operator tags Layer_N/Head_N as DEFECTIVE in OpenMetadata UI
2. Backend polls or receives webhook via /api/v1/webhooks/openmetadata
3. HeadMaskStore updates via /api/v1/openmetadata/sync-defects
4. Masks apply to next generation (zeroed via projection hooks)
5. Observable behavior change in output
```

---

## Hackathon Experience

Fast-paced and instructive. We built a working end-to-end system in a short timeframe:

- **Day 1**: Core tracing engine + OpenMetadata bootstrap
- **Day 2**: Dashboard + circuit discovery + quarantine flow
- **Day 3**: Testing, polish, and demo recording

### Key Challenges Overcome

1. **Hook registration**: Understanding PyTorch hook lifecycle and ensuring clean cleanup
2. **Two-level masking**: Implementing attention tensor masking + projection masking for effective ablation
3. **OpenMetadata integration**: Navigating the SDK for topology creation and lineage ingestion

### Key Takeaways

- Interpretability tooling can be both research-quality and operationally useful
- OpenMetadata provides a familiar governance framework for AI internals
- The gap between "interesting research" and "deployable system" is closable with the right abstractions

---

## Links Summary

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/FiscalMindset/Synapse-Graph |
| GitHub Pages | https://fiscalmindset.github.io/Synapse-Graph/ |
| YouTube Demo | https://youtu.be/idOJYh6TUC8 |
| Product Demo | https://youtu.be/b78Y7RwvYeU |