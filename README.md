# Synapse-Graph (AI Autopsy Engine)

<div align="center">

**Turn LLM internals into observable, governable infrastructure**

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-blue?style=for-the-badge)](https://fiscalmindset.github.io/Synapse-Graph/)
[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo-red?style=for-the-badge)](https://youtu.be/idOJYh6TUC8)
[![YouTube Product](https://img.shields.io/badge/YouTube-Product-red?style=for-the-badge)](https://youtu.be/b78Y7RwvYeU)

</div>

---

## Live Presentation

**HTML Presentation** — [Open in browser](https://fiscalmindset.github.io/Synapse-Graph/first_frame.html)

The presentation walks through the complete demo flow in a video-frame aesthetic (no scrolling, single screen per scene):

| Scene | Description |
|-------|-------------|
| [Intro](https://fiscalmindset.github.io/Synapse-Graph/first_frame.html) | Project overview, features, and graph visualization |
| [Architecture](https://fiscalmindset.github.io/Synapse-Graph/video_diagram.html) | Component flow: Dashboard → Neural Proxy → Tracer → OpenMetadata |
| [Tech Stack](https://fiscalmindset.github.io/Synapse-Graph/tech_stack.html) | Exact dependencies from pyproject.toml and package.json |
| [Live Demo](https://fiscalmindset.github.io/Synapse-Graph/video_scene.html) | Demo video showcasing circuit discovery and quarantine workflow |
| [OpenMetadata](https://fiscalmindset.github.io/Synapse-Graph/openmetadata_usage.html) | Governance plane: topology mapping and quarantine flow |
| [Project Status](https://fiscalmindset.github.io/Synapse-Graph/project_status.html) | Current capabilities and open research challenges |
| [Thank You](https://fiscalmindset.github.io/Synapse-Graph/last_frame.html) | Credits and links |

## Video Demos

<div align="center">

[![Demo Video Thumbnail](https://img.youtube.com/vi/idOJYh6TUC8/0.jpg)](https://youtu.be/idOJYh6TUC8)

**[Complete Demo Walkthrough](https://youtu.be/idOJYh6TUC8)** — 2-3 min walkthrough of the full circuit discovery → quarantine → re-run loop

**[Product Demo](https://youtu.be/b78Y7RwvYeU)** — Live run showing active heads, lineage, and quarantine enforcement

</div>

---

## The Problem

LLMs are powerful but opaque. Current observability stops at prompts, tokens, latency, and logs. They don't answer:

- *Which layers and heads were most active for this response?*
- *Can we trace a "thought path" through the network in a way operators can inspect?*
- *Can governance tools intervene on specific neural components?*

## The Solution

Synapse-Graph repurposes **OpenMetadata** as a governance and lineage system for transformer internals:

- Model → **Database**
- Transformer layers → **Tables**
- Attention heads → **Columns**
- High-activation paths → **Lineage edges**
- `DEFECTIVE` tag → **Runtime control signal** that masks a head during next generation

## The Impact

Turns model internals into observable, governable infrastructure. Instead of treating neural behavior as a black box, Synapse-Graph makes it **inspectable infrastructure** with familiar data-platform primitives.

---

## Quickstart

### Prerequisites

- Python `3.11` or `3.12`
- Node.js `20+`
- Optional: [Ollama](https://ollama.com) at `http://127.0.0.1:11434`
- Optional: [OpenMetadata](https://openmetadata.org) at `http://127.0.0.1:8585`

### 1. Backend

```bash
git clone https://github.com/FiscalMindset/Synapse-Graph.git
cd Synapse-Graph

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ./backend
cp backend/.env.example backend/.env

cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
# in a new terminal
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Dashboard: `http://localhost:3000`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Operator Dashboard                      │
│              Next.js + React + @xyflow/react               │
└─────────────────────────┬───────────────────────────────────┘
                        │ REST + SSE
┌─────────────────────────▼───────────────────────────────────┐
│                  Neural Proxy (FastAPI)                     │
│         Orchestrates generation + tracing + governance        │
└──────┬────────────────────┬────────────────────┬────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐    ┌────────────────┐    ┌─────────────────┐
│   Ollama     │    │  HF Tracer      │    │  OpenMetadata    │
│ (Preferred) │    │ (PyTorch hooks) │    │ (Governance)     │
│             │    │                │    │                  │
│ Generation  │    │ Attention      │    │ Topology        │
│             │    │ activation     │    │ Lineage          │
│             │    │ capture        │    │ Tags → Masks     │
└─────────────┘    └────────────────┘    └─────────────────┘
```

**Runtime flow:**

1. Operator submits prompt via dashboard
2. Backend runs generation (Ollama preferred) + parallel HF tracing
3. Tracer captures per-layer, per-head attention activations via `register_forward_hook`
4. Backend ingests lineage edges into OpenMetadata
5. Operator tags defective heads → backend converts tags to head masks
6. Subsequent generations run with masked heads zeroed out

---

## Tech Stack

### Backend
```toml
# backend/pyproject.toml
[project]
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
  "dependencies": {
    "next": "^15.2.0",
    "react": "^19.0.0",
    "@xyflow/react": "^12.4.4",
    "recharts": "^2.15.0",
    "lucide-react": "^0.468.0"
  }
}
```

---

## Core Capabilities

| Capability | Description |
|------------|-------------|
| **Circuit Discovery** | Ablation sweeps (single + pair) to find heads that materially change outputs |
| **Attention Tracing** | Hook-based per-layer, per-head activation capture via PyTorch |
| **OpenMetadata Governance** | Topology bootstrap, lineage ingestion, column-level tagging |
| **Runtime Masking** | `DEFECTIVE` tags → head masks applied to subsequent generations |
| **SSE Streaming** | Real-time progress updates for discovery and generation |
| **Evidence Quality** | Scoring system for trace confidence (high/medium/low) |

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/state` | GET | Current runtime state |
| `/api/v1/generate/stream` | POST | Stream generation with trace steps |
| `/api/v1/autopsy/discover_circuit` | POST | Discover hallucination circuits |
| `/api/v1/autopsy/causal` | POST | Run causal autopsy on a head |
| `/api/v1/openmetadata/bootstrap` | POST | Bootstrap OpenMetadata catalog |
| `/api/v1/openmetadata/sync-defects` | POST | Sync defective heads from OpenMetadata |
| `/api/v1/openmetadata/quarantine` | POST | Quarantine heads in OpenMetadata |
| `/api/v1/webhooks/openmetadata` | POST | Handle OpenMetadata webhooks |
| `/api/v1/governance/local-mask` | POST | Set local head mask |
| `/api/v1/hf/preload` | POST | Force-load HuggingFace tracer |

---

## OpenMetadata Usage

**Topology Bootstrap:**
```
Service: Synapse_Neural_Service
  └── Database: {Model_Name}
        └── Schema: Transformer_Graph
              ├── Table: Prompt_Ingress
              ├── Table: Response_Egress
              └── Table: Layer_N (one per layer)
                    └── Column: Head_N (one per head)
```

**Governance Loop:**
```
1. Operator tags Layer_N/Head_N as DEFECTIVE in OpenMetadata
2. Backend calls sync-defects → reads tags
3. HeadMaskStore updates → masks apply to next generation
4. Masked heads zeroed via projection hooks
```

---

## Repository Layout

```
Synapse-Graph/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app + endpoints
│   │   ├── inference.py     # Generation + tracing engine
│   │   └── om_client.py    # OpenMetadata client
│   └── tests/
│       ├── test_quarantine.py
│       └── test_discover_quarantine_integration.py
├── frontend/
│   ├── app/                # Next.js app router
│   ├── components/        # Dashboard, graph, charts
│   └── lib/               # API client
└── first_frame.html       # GitHub Pages presentation
```

---

## Environment Configuration

```bash
# backend/.env
SYNAPSE_OPENMETADATA_ENABLED=true
SYNAPSE_OPENMETADATA_HOST=http://127.0.0.1:8585
SYNAPSE_OLLAMA_MODEL=qwen2.5:3b-instruct
SYNAPSE_HF_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
SYNAPSE_PRELOAD_SHADOW_MODEL=true
```

---

## Demo Workflow

1. **Start**: Boot dashboard, verify Ollama or HF fallback
2. **Trace**: Submit prompt → watch synapse graph light up
3. **Discover**: Select hallucination token → run circuit discovery
4. **Quarantine**: Tag defective head → sync to backend
5. **Verify**: Re-run prompt → show masked heads in output

---

## License

MIT

---

<p align="center">
  <a href="https://github.com/FiscalMindset/Synapse-Graph">GitHub</a> ·
  <a href="https://fiscalmindset.github.io/Synapse-Graph/first_frame.html">Presentation</a> ·
  <a href="https://youtu.be/idOJYh6TUC8">YouTube Demo</a>
</p>