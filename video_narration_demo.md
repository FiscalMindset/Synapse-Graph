# Video Narration (Demo)

Purpose: Voiceover script and recorder actions for the Synapse-Graph demo. The presenter (recorder) follows the "Recorder actions" steps while the voice actor reads the short "Voiceover" lines aloud. Keep lines short, conversational, and present-tense.

Recording notes
- Resolution: 1920x1080 (1080p). Frame rate: 30fps (60fps optional).
- Cursor visible; highlight clicks with a subtle click sound (optional). Pause 0.6–1s after each click to let the narrator describe the action.
- Voice style: friendly, confident, concise. Pace: ~150 wpm. Emphasize keywords: "discovery", "circuit", "quarantine", "OpenMetadata".

Scene 1 — Intro (first_frame.html)
- Recorder actions:
  - Open `first_frame.html` full-screen.
  - Let the hero visual sit for 3–4s, move cursor slowly over the schematic, then hover over the features list. Pause.
  - Click the "Start demo →" CTA.
- Voiceover:
  - "Hi — I'm demonstrating Synapse‑Graph: automated causal‑circuit discovery with OpenMetadata governance and differential traces."
  - "I'll run a discovery, inspect the explanation traces, and quarantine a circuit in the metadata control plane." (Pause)
  - "Let's start." (Short pause while clicking.)
- Suggested duration: 10–12s total.

Scene 2 — Architecture (video_diagram.html)
- Recorder actions:
  - After CTA click, let `video_diagram.html` load.
  - Slowly pan the Mermaid diagram: point at the Frontend → Neural Proxy → Shadow Tracer → OpenMetadata flow.
  - Hover each right-side info card (Frontend, Neural Proxy, Shadow Tracer, OpenMetadata) for 2–3s each.
- Voiceover:
  - "This is the runtime architecture. The Next.js dashboard drives a FastAPI neural proxy." (Pause while hovering Frontend → Neural Proxy)
  - "The proxy coordinates generation and a Shadow Tracer that captures attention traces from the model." (Pause while hovering Tracer)
  - "OpenMetadata stores the topology and tags used for governance and enforcement." (Pause while hovering OpenMetadata)
- Suggested duration: 18–25s.

Scene 3 — Tech Stack (tech_stack.html)
- Recorder actions:
  - Click the "Tech" nav to open `tech_stack.html`.
  - Scroll to the backend and frontend stack cards; highlight the backend list briefly, then the frontend list.
  - Optionally open the manifest snippet on the right and scroll through the excerpt.
- Voiceover:
  - "Key components: FastAPI and PyTorch for the backend; Next.js and React for the dashboard." (Pause)
  - "We use the OpenMetadata ingestion SDK for mapping layers and heads into the metadata plane." (Short pause)
- Suggested duration: 12–16s.

Scene 4 — OpenMetadata usage (openmetadata_usage.html)
- Recorder actions:
  - Click "OpenMetadata usage"; show the topology mermaid.
  - Hover the path from model → table (layer) → column (head). Point at the tag arrow `DEFECTIVE / QUARANTINED` and the arrow into the backend enforcement step.
  - Scroll to the Governance flow list and pause on the runtime enforcement line.
- Voiceover:
  - "We model each transformer layer as a table and each head as a column in OpenMetadata." (Pause)
  - "When an operator tags a head as `DEFECTIVE` or `QUARANTINED`, the backend updates runtime masks so subsequent generations respect the quarantine." (Pause)
- Suggested duration: 16–22s.

Scene 5 — Project status & roadmap (project_status.html)
- Recorder actions:
  - Click "Project status". Slow scroll through the "Completed checkpoints" list; linger on tests/CI and overlay traces items.
  - Scroll to the "Future goals & growth roadmap"; highlight a couple of bullets.
- Voiceover:
  - "Most core functionality is implemented: webhook handling, OpenMetadata tagging, discovery flows, and automated tests." (Pause)
  - "Planned next steps include deeper ablation sweeps, optional OpenMetadata mocks for CI, and UX polish." (Short pause)
- Suggested duration: 12–18s.

Scene 6 — Demo player (video_scene.html)
- Recorder actions:
  - Open `video_scene.html`.
  - Paste the recorded MP4 URL into the input (or use a local blob URL). Click "Load" and then play the video.
  - If the video contains a capture of the live dashboard run, cue the segment showing the discovery run. Otherwise, play the recorded clip as-is.
- Voiceover:
  - "Now I'll load the recorded demo clip so you can see Synapse‑Graph in action." (Pause while pasting URL)
  - "Playing the capture now." (Hold while the clip plays.)
- Suggested duration: depends on clip length; cue narration to align to visible steps in the recorded capture.

Scene 7 — Live dashboard: discovery run (live app)
- Preconditions: the Next.js dashboard backend (Neural Proxy) and model runtime are running so you can trigger a live discovery. If not available, this can be demonstrated in the recorded clip from Scene 6.
- Recorder actions (live):
  - Open the Synapse‑Graph dashboard.
  - Select a short prompt or example input, choose the model/topology if required, and click "Run discovery" or equivalent action.
  - Wait for ranking + ablation steps to complete — show any progress indicator.
  - When results appear: enlarge the discovered circuit, toggle the ablated trace overlay (baseline vs ablated), and show the `combined_causal_effect` metric.
  - Click the "QUARANTINE CIRCUIT IN OPENMETADATA" action; confirm the modal and show the resulting tag in OpenMetadata (or show the webhook/mask update flow if visible).
- Voiceover (concise, step-by-step):
  - "I'm triggering a discovery run on a short example prompt. The system first ranks candidate heads and runs focused ablation sweeps to estimate causal effects." (Pause while discovery runs)
  - "Here is the discovered circuit and the combined causal‑effect score — darker overlays show the ablated trace compared to baseline." (Pause while toggling overlay)
  - "I'll quarantine the discovered circuit now. That adds a `QUARANTINED` tag in OpenMetadata and the runtime snapshot updates masks for future generations." (Pause while confirming)
- Suggested duration: 25–45s depending on discovery runtime; keep narration aligned to visible state changes.

Scene 8 — Wrap-up & CTA
- Recorder actions:
  - Return to `first_frame.html` or show the README/GitHub repo page.
  - Show links and the repo URL.
- Voiceover:
  - "That's Synapse‑Graph: discovery, differential traces, and governance via OpenMetadata."
  - "Find the code, docs, and demo assets in the repository linked on screen. Thanks for watching." (End)
- Suggested duration: 8–12s.

Voice actor cues & timing
- Keep each sentence short; breathe between clauses (0.3–0.6s). Use slightly slower pacing when describing technical steps.
- When the recorder is clicking or a visual change occurs, pause briefly and then describe the result.

Optional additions (ask me if you'd like)
- I can produce an SRT-style timestamped transcript keyed to scene durations.
- I can convert the script to a single-speaker TTS-ready file or produce per-line durations for lip-sync.

---

File created: `video_narration_demo.md`

Next: I'll mark the narration steps complete in the todo list. If you'd like different narration tone (third‑person, more/less technical, or longer/shorter lines), tell me which scenes to modify and I will update the file.