# 3-Minute Demo Video Script — Synapse-Graph (AI Autopsy Engine)

Total length: ~2:30–3:00

Diagram (open for the About segment)

Open `video_diagram.html` in a browser and display it at 0:08–0:40 while you narrate the architecture. The HTML file uses a Mermaid diagram and lists the exact backend/frontend dependencies taken from the repository files so judges see the real tech snapshot.

0:00–0:08 — Title & Hook (8s)
- On-screen: Project title, your name/handle, one-line hook: "Turn LLM internals into a governance plane."
- Narration: "Hi, I'm <Your Name>. This is Synapse-Graph — an LLM glassbox that maps neural activity to OpenMetadata."

0:08–0:40 — About the project (32s)
- On-screen: short architecture diagram or screenshot of dashboard.
- Narration: "Synapse-Graph captures attention activations from a running transformer, ingests them into OpenMetadata as model layers and head columns, and visualizes active neural routes in a dashboard. We then run causal ablation sweeps to identify small head sets that cause hallucinations and let operators quarantine them via metadata tags."

Note: During this segment open `video_diagram.html` (browser) and briefly highlight the following nodes: Dashboard (Next.js), FastAPI Neural Proxy, Shadow Tracer (PyTorch+HF), Generation Engine (Ollama/HF), OpenMetadata, HeadMaskStore.

0:40–1:05 — Tech stack & architecture (25s)
- On-screen: list of components (FastAPI backend, PyTorch tracer, Next.js UI, OpenMetadata) with icons; quick flow arrow animation.
- Narration: "Built with Python/FastAPI and a PyTorch shadow tracer, the proxy prefers local Ollama for generation and uses a Hugging Face model for topology and tracing. The frontend is Next.js + React Flow; OpenMetadata stores topology, lineage, and governance tags."

1:05–2:00 — Live demo (55s)
- Action: Screen-record the dashboard. Submit a short prompt (e.g., "List inventions or practical devices Albert Einstein is credited with").
- On-screen: show token stream, synapse graph lighting up, activation panel.
- Narration (short punchlines as actions occur):
  - "I submit a prompt — the proxy streams tokens and the tracer captures attention activity." (show stream)
  - "Here’s a high-activation route that correlates with a hallucination token." (highlight path)
  - "I run the discovery/autopsy — it ranks heads and runs ablation sweeps to measure causal effect." (open sweep results)
  - "I quarantine the discovered circuit — this pushes `DEFECTIVE` tags to OpenMetadata and updates runtime masks." (click quarantine)

2:00–2:30 — Re-run & results (30s)
- Action: Re-run the same prompt.
- On-screen: baseline vs ablated overlay (differential traces) and changed output text.
- Narration: "After quarantine, the same prompt produces a different output — the differential overlay shows reduced activation along the previously causal route."

2:30–2:50 — Learning & next steps (20s)
- Narration: "We learned how to map interpretability outputs to a governance plane, and how lightweight metadata operations can control runtime behavior. Next steps: add deeper pair/triple sweeps, automate suggested quarantines, and provide audit logs in OpenMetadata."

2:50–3:00 — Call to action (10s)
- On-screen: GitHub link and contact.
- Narration: "Try it locally — see the README for quickstart, and open an issue or PR at the repo. Thanks!"

Recording tips
- Use a single-screen capture, 1080p, and crop to the dashboard area.
- Record voiceover in a quiet room; keep sentences short and time each segment.
- Use the script as a teleprompter; practice once before recording to stay within 3 minutes.

Notes for editing
- Trim pauses and speed up non-essential screen parts (e.g., long model loads) to keep within time.
- Emphasize the discovery → quarantine → behavioral change loop.
