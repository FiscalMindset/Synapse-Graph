# Video Narration — Synapse-Graph Demo

This file contains the spoken narration for each demo scene. Each section links to the corresponding HTML scene so you can open the page while reading aloud.

- Open the page in a browser and follow the short cue lines. Keep each narration ~15–40 seconds per scene for a ~3 minute total.

---

## 0 — Intro (First frame)
File: [first_frame.html](first_frame.html#intro)

Cue: "Hi, I'm Vicky Kumar. This is Synapse-Graph — a small system that discovers causal circuits responsible for hallucinations, quarantines defective attention heads via OpenMetadata, and provides differential traces for debugging. I'll walk you through architecture, a live discovery run, and how governance updates take effect at runtime."

Notes: Pause while the intro card is visible (3–4s), then click Start demo.

---

## 1 — Architecture (Runtime diagram)
File: [video_diagram.html](video_diagram.html#architecture)

Cue: "Here's the runtime architecture. The frontend dashboard controls discovery and shows differential overlays. The Neural Proxy (FastAPI) orchestrates generation and the Shadow Tracer instruments the model to emit attention traces. The HeadMaskStore keeps masking state and synchronizes quarantines with OpenMetadata. During discovery we rank heads, sweep single and pair ablations, and surface the best causal circuit."

Notes: Use pointer to highlight the tracer → HeadMaskStore → OpenMetadata arrows while you speak.

---

## 2 — Tech Stack (Concise)
File: [tech_stack.html](tech_stack.html#tech)

Cue: "Tech stack at a glance: Python backend using FastAPI, PyTorch tracing, and OpenMetadata ingestion; Next.js + React frontend with a graph visualizer; CI runs `pytest` and TypeScript checks. The repo constrains Python to 3.11–3.12 for reproducibility."

Notes: Brief, punchy. Mention `openmetadata-ingestion` and `transformers` when showing package list.

---

## 3 — Live discovery demo (Video preview)
File: [video_scene.html](video_scene.html#video)

Cue: "Now I'll run a discovery for a selected hallucination token. The system ranks heads by activation correlation, then runs single- and pair-head ablations while capturing ablated traces. The best circuit and combined causal effect appear in the result pane, and ablated traces are returned for UI overlays so you can compare baseline vs ablated activations."

Notes: If showing a recorded clip, paste the clip URL into the player and play during narration.

---

## 4 — OpenMetadata & Governance
File: [openmetadata_usage.html](openmetadata_usage.html#om)

Cue: "OpenMetadata stores model topology and serves as our governance coordination plane. When we mark a head as `DEFECTIVE` or `QUARANTINED`, the webhook handler updates the HeadMaskStore in memory so subsequent generations immediately apply the mask without restarting the runtime. Tagging prefers column-level context and falls back to table-level when needed."

Notes: Mention the retry/backoff behavior briefly as an engineering robustness note.

---

## 5 — Project Status & Next Steps
File: [project_status.html](project_status.html#status)

Cue: "Status: discovery and quarantine flows implemented, unit and integration tests added, CI runs typechecks and pytest. Next steps include adding an OpenMetadata mock for CI and UX polish for the differential overlays."

Notes: Close the demo by inviting questions and pointing reviewers to `explain.md` and `video_script.md` in the repo.

---

## Quick timings (approx)

- Intro: 20s
- Architecture: 35s
- Tech stack: 20s
- Live discovery / video preview: 60s
- OpenMetadata governance: 30s
- Status + close: 15s

Total ≈ 2.5–3 minutes depending on small pauses.
