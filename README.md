# Synapse-Graph (AI Autopsy Engine)

**Turn LLM internals into observable, governable infrastructure**

Live Presentation: https://fiscalmindset.github.io/Synapse-Graph/first_frame.html
YouTube Demo: https://youtu.be/idOJYh6TUC8
Product Demo: https://youtu.be/b78Y7RwvYeU

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Operator Dashboard                       │
│              Next.js + React + @xyflow/react               │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST + SSE
┌─────────────────────────▼───────────────────────────────────┐
│                  Neural Proxy (FastAPI)                    │
│         Orchestrates generation + tracing + governance      │
└──────┬────────────────────┬────────────────────┬───────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐    ┌────────────────┐    ┌─────────────────┐
│   Ollama     │    │  HF Tracer      │    │  OpenMetadata   │
│ (Preferred) │    │ (PyTorch hooks) │    │  (Governance)   │
│             │    │                │    │                 │
│ Generation  │    │ Attention      │    │ Topology        │
│             │    │ activation     │    │ Lineage         │
│             │    │ capture        │    │ Tags → Masks    │
└─────────────┘    └────────────────┘    └─────────────────┘
```

## OpenMetadata Topology

```
Service: Synapse_Neural_Service
  └── Database: {Model_Name}
        └── Schema: Transformer_Graph
              ├── Table: Prompt_Ingress (columns: Prompt_Text, Token_Count)
              ├── Table: Response_Egress (columns: Response_Text)
              └── Table: Layer_N
                    └── Column: Head_N (per head, FLOAT)

Lineage: Prompt_Ingress → Layer_1 → ... → Layer_N → Response_Egress
```

## The Problem → Solution → Impact

```
┌──────────────────────────────────────────────────────────────┐
│  PROBLEM                                                     │
│  LLMs are powerful but opaque                                │
│  Current observability stops at prompts, tokens, latency       │
│  Cannot inspect which heads produced a hallucination         │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  SOLUTION                                                    │
│  Repurpose OpenMetadata as governance plane for neural internals│
│                                                              │
│  Model      → Database                                        │
│  Layer      → Table                                           │
│  Head       → Column                                          │
│  Activations→ Lineage edges                                    │
│  DEFECTIVE  → Runtime control signal (head mask)                 │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  IMPACT                                                     │
│  Answer "which heads produced this hallucination?"            │
│  Operationalize mitigation at head level                     │
│  Make AI interpretability an operational control plane      │
└──────────────────────────────────────────────────────────────┘
```

## Tech Stack

```
┌─────────────────────────────────────────────────────────────┐
│  BACKEND                                                     │
│  Python 3.11+                                               │
│  ├── fastapi >= 0.115.0         (HTTP API)                   │
│  ├── torch >= 2.4.0             (Tracing engine)            │
│  ├── transformers >= 4.46.0    (Model interface)          │
│  ├── openmetadata-ingestion >= 1.12.0 (Metadata SDK)        │
│  ├── httpx >= 0.28.0            (REST client)             │
│  ├── pydantic-settings >= 2.7.0 (Config)                   │
│  ├── uvicorn[standard] >= 0.32.0 (Server)                │
│  └── accelerate >= 1.1.0         (HF acceleration)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  FRONTEND                                                    │
│  ├── next ^15.2.0              (Framework)                 │
│  ├── react ^19.0.0             (UI)                       │
│  ├── @xyflow/react ^12.4.4      (Graph visualization)      │
│  ├── recharts ^2.15.0            (Charts)                   │
│  └── lucide-react ^0.468.0       (Icons)                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Capabilities

```
┌─────────────────────────────────────────────────────────────┐
│  CIRCUIT DISCOVERY                                           │
│  ├── Single-head ablation sweeps                              │
│  ├── Pair-head ablation sweeps (O(n²))                       │
│  └── Causal effect scoring per head                         │
├─────────────────────────────────────────────────────────────┤
│  ATTENTION TRACING                                           │
│  ├── register_forward_hook on attention modules              │
│  ├── Per-layer, per-head activation capture                 │
│  └── Top-K heads & positions tracking                     │
├─────────────────────────────────────────────────────────────┤
│  RUNTIME MASKING                                            │
│  ├── Two-level: attention tensor + projection masking       │
│  ├── HeadMaskStore for runtime state                       │
│  └── DEFECTIVE tags → zeroed outputs on next generation    │
├─────────────────────────────────────────────────────────────┤
│  SSE STREAMING                                              │
│  ├── Real-time trace step updates                           │
│  ├── Discovery progress streaming                         │
│  └── Generation with live lineage                         │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

```
┌─────────────────────────────────────────────────────────────┐
│  GENERATION                                                 │
│  POST /api/v1/generate              Full response          │
│  POST /api/v1/generate/stream       SSE with trace steps    │
├─────────────────────────────────────────────────────────────┤
│  AUTOPSY / CIRCUIT DISCOVERY                                │
│  POST /api/v1/autopsy/discover_circuit         Main discovery │
│  POST /api/v1/autopsy/discover_circuit/stream  SSE streaming │
│  POST /api/v1/autopsy/causal                    Causal check  │
├─────────────────────────────────────────────────────────────┤
│  OPENMETADATA                                              │
│  POST /api/v1/openmetadata/bootstrap       Bootstrap catalog│
│  POST /api/v1/openmetadata/sync-defects  Sync tags to masks │
│  POST /api/v1/openmetadata/quarantine   Quarantine heads  │
│  POST /api/v1/webhooks/openmetadata    Webhook handler   │
├─────────────────────────────────────────────────────────────┤
│  GOVERNANCE                                               │
│  POST /api/v1/governance/local-mask      Set head mask    │
│  POST /api/v1/governance/clear-local-masks  Clear masks  │
│  POST /api/v1/hf/preload                 Load HF tracer │
└─────────────────────────────────────────────────────────────┘
```

## Quickstart

```
# 1. Backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ./backend
cp backend/.env.example backend/.env
cd backend && python -m uvicorn app.main:app --reload --port 8000

# 2. Frontend
cd frontend && npm install && npm run dev
# Dashboard: http://localhost:3000
```

## Demo Workflow

```
┌─────��───────────────────────────────────────────────────────┐
│  1. START                                                  │
│     Boot dashboard                                         │
│     Verify backend: "Ollama live" or "HF fallback"          │
├─────────────────────────────────────────────────────────────┤
│  2. TRACE                                                  │
│     Submit prompt                                          │
│     Watch synapse graph light up with active edges          │
├─────────────────────────────────────────────────────────────┤
│  3. DISCOVER                                               │
│     Select hallucination token                              │
│     Run circuit discovery (single + pair ablation)          │
├─────────────────────────────────────────────────────────────┤
│  4. QUARANTINE                                             │
│     Click "Quarantine" on discovered head                   │
│     Push DEFECTIVE tags to OpenMetadata                    │
├─────────────────────────────────────────────────────────────┤
│  5. VERIFY                                                 │
│     Re-run prompt                                          │
│     Show masked heads count increase                        │
│     Show changed output (baseline vs ablated)             │
└─────────────────────────────────────────────────────────────┘
```

## Governance Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Operator tags Layer_N/Head_N as DEFECTIVE in OpenMetadata  │
│                            │                               │
│                            ▼                               │
│  Backend reads tags via /api/v1/openmetadata/sync-defects     │
│                            │                               │
│                            ▼                               │
│  HeadMaskStore updates → masks apply to next generation      │
│                            │                               │
│                            ▼                               │
│  Masked heads zeroed via PyTorch projection hooks          │
│                            │                               │
│                            ▼                               │
│  Observable behavior change in next generation            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Repo Structure

```
Synapse-Graph/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI + endpoints
│   │   ├── inference.py     # Generation + tracing engine
│   │   └── om_client.py    # OpenMetadata client
│   └── tests/
│       ├── test_quarantine.py
│       └── test_discover_quarantine_integration.py
├── frontend/
│   ├── app/                 # Next.js app router
│   ├── components/          # Dashboard, graph, charts
│   └── lib/                 # API client
├── first_frame.html          # GitHub Pages presentation
├── video_diagram.html       # Architecture scene
├── tech_stack.html         # Tech stack scene
├── video_scene.html       # Demo video scene
├── openmetadata_usage.html # Governance scene
├── project_status.html    # Status + gaps scene
└── last_frame.html       # Thank you scene
```

## Links

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/FiscalMindset/Synapse-Graph |
| Presentation | https://fiscalmindset.github.io/Synapse-Graph/first_frame.html |
| YouTube Demo | https://youtu.be/idOJYh6TUC8 |
| Product Demo | https://youtu.be/b78Y7RwvYeU |

---

MIT License