# Video Narration — Frontend Demo

Purpose: A focused voiceover script and recorder-action checklist for recording *only the frontend dashboard* (no backend or other static frames). Use this when you (the recorder) will interact with the live dashboard or play a recorded clip of the frontend UI. The voice actor should read the short first-person lines while the recorder performs the listed actions.

Recording preconditions
- Prefer: run the Next.js app locally (http://localhost:3000) so the live UI is available. If not, play a captured clip in the `video_scene.html` player.
- Video: 1920x1080 (1080p), 30fps. Cursor visible. Use a short pause (0.6–1s) after clicks.
- Voice: first-person present tense, conversational, ~150 wpm. Pause 0.3–0.6s after each sentence.

Accuracy constraint
- Do not narrate or assert backend behavior, logs, or metadata propagation unless that information is clearly visible in the frontend UI you are recording. Only describe what the dashboard visibly shows (buttons clicked, success toasts, visible indicators). If a backend console is open on-screen and you explicitly want it included, treat it as on-screen evidence — otherwise do not reference or claim off-screen effects.

Scene 1 — Open dashboard (Landing)
- Recorder actions:
  - Open the dashboard at `http://localhost:3000` (or open the deployed URL/recorded clip) and wait for the main view to finish loading.
  - Make sure the Synapse graph and control panel are visible.
- Voiceover:
  - "I'm opening the Synapse‑Graph dashboard to run a focused frontend demo." (Pause)
  - "I'll trigger an automated discovery, inspect the discovered circuit, and quarantine it from the UI." (Short pause)
- Timing: 6–9s

Scene 2 — Enter prompt & start discovery
- Recorder actions:
  - In the input pane, type a short example prompt (e.g., "Translate to French: Hello world") or pick a provided example.
  - Click the button labeled `Run discovery` (or the dashboard's equivalent) to start the analysis.
- Voiceover:
  - "I enter a short example prompt and start a discovery run." (Click)
  - "The dashboard first ranks candidate heads and runs ablations to estimate causal effect." (Short pause while the UI shows the progress bar)
- Timing: 8–15s (depends on UI progress animation)

Scene 3 — Watch ranking & ablation progress
- Recorder actions:
  - Focus on the results pane. Let the camera show the ranking table / candidate list as it populates.
  - If the UI shows a progress timeline, let it run until the top candidate appears.
- Voiceover:
  - "Here the UI shows ranked candidate heads and the ablation sweep as it runs." (Pause)
  - "You can see progress and intermediate rankings as the discovery narrows in." (Short pause)
- Timing: 8–12s

Scene 4 — Inspect discovered circuit
- Recorder actions:
  - Click to open the top discovered circuit result.
  - Expand the circuit details: show layer/head groupings and the `combined_causal_effect` metric.
  - Zoom or pan the synapse graph visualization so the discovered circuit is centered and clearly visible.
- Voiceover:
  - "This is the discovered circuit — the set of heads the system found most causally relevant." (Pause while opening details)
  - "Notice the combined causal‑effect score here — it summarizes the circuit's influence." (Short pause)
- Timing: 10–18s

Scene 5 — Differential traces (baseline vs ablated)
- Recorder actions:
  - Toggle the trace overlay control to show baseline traces, then switch to the ablated overlay (or show differential overlay if available).
  - Animate or scrub the token timeline so the baseline and ablated traces are visible in sequence.
- Voiceover:
  - "I toggle the differential overlay to compare baseline traces against the ablated traces." (Pause while toggling)
  - "The overlay shows how ablation changes activations over tokens — this makes causal effects visually obvious." (Short pause)
- Timing: 12–20s

Scene 6 — Quarantine from the frontend
- Recorder actions:
  - Click the action button labeled `QUARANTINE CIRCUIT IN OPENMETADATA` (or the UI's quarantine action).
  - Confirm the modal dialog (click Confirm / Yes) if shown. Focus the recording on the modal confirmation and the subsequent UI feedback (success toast or status indicator).
Voiceover:
  - "I'll perform the quarantine action from the UI." (Click)
  - "Confirming now — the dashboard shows a success message confirming the action." (Pause while confirming)
Note: Only narrate what the frontend visibly confirms. Do not claim that the tag has been applied in off-screen metadata or that runtime masks took effect unless you can show an on-screen indicator or replay demonstrating that effect.
Timing: 10–16s

Scene 7 — Show mask/status and replay effect (optional)
- Recorder actions:
  - Only perform this step if the frontend UI can demonstrably show the effect (for example, an on-screen masked‑heads indicator or an immediate replay that visibly changes activations).
  - If available, run a short generation or replay and point the camera at the visible UI change.
Voiceover:
  - "If visible here, point out the masked‑heads indicator or replay result — describe only what you observe on-screen." (Pause)
Timing: 8–14s

Scene 8 — Wrap and CTA (repo link)
- Recorder actions:
  - Return to the top navigation and show the repo link or README link on screen.
  - Pause for 3–4s on the repo link.
- Voiceover:
  - "That's the frontend demo: discovery, differential traces, and quarantine — all from the dashboard." (Short pause)
  - "Find the code and demo assets in the repository linked on screen. Thanks for watching." (End)
- Timing: 6–10s

Voice actor cues & tone
- Read in first‑person present tense ("I'm doing X"). Keep sentences short.
- Use a confident, explanatory tone; slow slightly for technical phrases ("combined causal‑effect", "differential traces").
- Leave 0.3–0.6s silence after each sentence; 0.6–1s after each recorded click or large visual change.

Optional outputs I can produce next
- A timed SRT file mapping the voice lines to approximate timestamps.
- A condensed micro‑script (one line per action) for quick reads.

---

File: `video_narration_frontend.md` (created)

If you want me to overwrite `video_narration_demo.md` with this frontend-only script instead, tell me and I'll replace it.