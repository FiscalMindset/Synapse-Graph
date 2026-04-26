# Project Submission Form — Synapse-Graph (AI Autopsy Engine)

---

## Project Description

Synapse-Graph transforms a running transformer model into an observable, governable system by:

- **Capturing per-token, per-layer, per-head attention activations** via a hook-instrumented PyTorch "shadow" tracer using `register_forward_hook`
- **Translating activations into OpenMetadata entities** (model → database → schema → layer tables → head columns) and ingesting lineage edges for active neural routes
- **Running causal ablation sweeps** (single-head and pair-head) to surface small head sets that materially change outputs for a target hallucination token
- **Closing the governance loop**: operators tag defective heads in OpenMetadata (`DEFECTIVE`/`QUARANTINED`), and the backend converts those tags to runtime head masks applied to subsequent generations

---

## Links

| Resource | URL |
|----------|-----|
| **GitHub** | https://github.com/FiscalMindset/Synapse-Graph |
| **GitHub Pages** | https://fiscalmindset.github.io/Synapse-Graph/ |
| **HTML Presentation** | https://fiscalmindset.github.io/Synapse-Graph/first_frame.html |
| **YouTube Complete Demo** | https://youtu.be/idOJYh6TUC8 |
| **YouTube Product Demo** | https://youtu.be/b78Y7RwvYeU |

---

## Problem

LLMs are powerful but operationally opaque. Current observability tools stop at prompts, tokens, latency, and logs. They cannot answer:

- Which layers and heads were most active for this response?
- Can we trace a "thought path" through the network?
- Can governance tools intervene on specific neural components instead of blocking whole prompts?

---

## Solution

Synapse-Graph repurposes **OpenMetadata** as a governance and lineage system for transformer internals:

| Neural Component | OpenMetadata Entity |
|-----------------|---------------------|
| Model | Database |
| Transformer Layer | Table |
| Attention Head | Column |
| High-activation path | Lineage Edge |
| DEFECTIVE tag | Runtime control signal |

---

## Impact

- Teams can answer "which heads produced this hallucination?"
- Operationalize mitigation at the head level rather than blocking the whole prompt
- Makes AI interpretability into an **operational control plane**, not just static visualization

---

## Tech Stack

### Backend
```toml
# Exact dependencies from backend/pyproject.toml
requires-python = ">=3.11,<3.13"

dependencies = [
    "fastapi>=0.115.0",
    "torch>=2.4.0",
    "transformers>=4.46.0",
    "openmetadata-ingestion>=1.12.0",
    "httpx>=0.28.0",
    "pydantic-settings>=2.7.0",
    "uvicorn[standard]>=0.32.0",
    "accelerate>=1.1.0",
    "cachetools>=5.3.0",
]
```

### Frontend
```json
{
  "next": "^15.2.0",
  "react": "^19.0.0",
  "@xyflow/react": "^12.4.4",
  "recharts": "^2.15.0",
  "lucide-react": "^0.468.0"
}
```

---

## Architecture (Runtime Flow)

```
1. User submits prompt via Dashboard
2. Backend runs generation (Ollama preferred) + parallel HF tracing
3. Tracer captures per-layer, per-head activations via PyTorch hooks
4. Backend ingests lineage edges into OpenMetadata
5. Operator tags defective heads in OpenMetadata
6. Backend converts tags to head masks via HeadMaskStore
7. Subsequent generations run with masked heads zeroed out
```

---

## Core Capabilities

| Capability | Implementation |
|------------|----------------|
| **Circuit Discovery** | Ablation sweeps: single-head + pair-head combinations |
| **Attention Tracing** | `register_forward_hook` on attention modules |
| **Head Masking** | Two-level: attention tensor + projection masking |
| **Evidence Quality** | Scoring system (high/medium/low confidence) |
| **Lineage** | Prompt → Layer → Head → Token edges in OpenMetadata |
| **Governance** | Column-level tagging → runtime head masks |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/generate/stream` | POST | Stream generation with trace steps |
| `/api/v1/autopsy/discover_circuit` | POST | Discover hallucination circuits |
| `/api/v1/autopsy/discover_circuit/stream` | POST | Stream discovery progress |
| `/api/v1/autopsy/causal` | POST | Run causal autopsy on a head |
| `/api/v1/openmetadata/bootstrap` | POST | Bootstrap OpenMetadata catalog |
| `/api/v1/openmetadata/sync-defects` | POST | Sync defective heads from OpenMetadata |
| `/api/v1/openmetadata/quarantine` | POST | Quarantine heads in OpenMetadata |
| `/api/v1/webhooks/openmetadata` | POST | Handle OpenMetadata webhooks |
| `/api/v1/governance/local-mask` | POST | Set local head mask |
| `/api/v1/hf/preload` | POST | Force-load HuggingFace tracer |

---

## OpenMetadata Usage

### Topology Bootstrap
- Creates synthetic entities: Service → Database → Schema → Tables (layers) → Columns (heads)
- Classification: `SynapseQuarantine` with tag `DEFECTIVE` (color: #39FF14)

### Lineage Ingestion
- Creates edges: `Prompt_Ingress` → `Layer_1` → ... → `Layer_N` → `Response_Egress`
- Stores activation path in SQL query field

### Governance Flow
1. Operator tags `Layer_N/Head_N` as `DEFECTIVE` in OpenMetadata UI
2. Backend polls or receives webhook → reads tags
3. `HeadMaskStore` updates → masks apply to next generation
4. Masked heads zeroed via projection hooks

---

## Demo Commands

```bash
# Terminal 1: Backend
cd Synapse-Graph
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ./backend
cd backend
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

Dashboard: `http://localhost:3000`

---

## Demo Steps (60-90 seconds)

1. **Start**: Boot dashboard, verify backend reports "Ollama live" or "HF fallback"
2. **Trace**: Submit prompt → watch synapse graph light up with active edges
3. **Discover**: Enter hallucination token → run circuit discovery
4. **Quarantine**: Click "Quarantine" on discovered head → push tags to OpenMetadata
5. **Verify**: Re-run prompt → show masked heads in metrics panel

---

## What Judges Should Notice

- Mapping of neural internals to metadata (tables/columns) in OpenMetadata
- Concrete causal signal: ablation removes/alters hallucinated token
- Simple governance loop: metadata tag → runtime mask → observable behavior change

---

## Project Status

### Completed
- Hook-based attention capture with PyTorch
- Causal circuit discovery via ablation sweeps
- OpenMetadata topology bootstrap and lineage ingestion
- Head masking runtime
- Dashboard with graph visualization
- SSE streaming for progress updates

### Open Challenges (Research Gaps)
- Causality proof (correlation vs causation)
- Ground-truth validation datasets
- Scalability for larger models (O(n²) pair ablation)
- Token-level vs layer-level tracing

---

## Repo Structure

```
Synapse-Graph/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── inference.py     # Generation + tracing
│   │   └── om_client.py    # OpenMetadata client
│   └── tests/
│       ├── test_quarantine.py
│       └── test_discover_quarantine_integration.py
├── frontend/
│   ├── app/                 # Next.js app router
│   ├── components/          # Dashboard, graph, charts
│   └── lib/                # API client
└── first_frame.html         # GitHub Pages presentation
```

---

## Notes

- Demo video: https://youtu.be/idOJYh6TUC8
- Product demo: https://youtu.be/b78Y7RwvYeU
- Presentation: https://fiscalmindset.github.io/Synapse-Graph/
- Default model: `Qwen/Qwen2.5-1.5B-Instruct` (HuggingFace) or `qwen2.5:3b-instruct` (Ollama)