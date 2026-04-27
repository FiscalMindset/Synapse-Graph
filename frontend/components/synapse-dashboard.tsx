"use client";

import { startTransition, useDeferredValue, useEffect, useState } from "react";
import { Orbit, Radio, ShieldAlert, Sparkles } from "lucide-react";

import { ActivationChart } from "@/components/activation-chart";
import { ConsoleLog } from "@/components/console-log";
import { SynapseGraph } from "@/components/synapse-graph";
import {
  clearLocalHeadMasks,
  fetchState,
  preloadHF,
  postDiscoverCircuit,
  postQuarantineCircuit,
  streamDiscoverCircuit,
  streamQuarantineCircuit,
  setLocalHeadMask,
  streamGeneration,
  syncOpenMetadataDefects,
  fetchCatalogDetail,
  fetchParsedEvidence,
} from "@/lib/api";
import type {
  AttentionTrace,
  LayerActivation,
  LogEntry,
  ModelTopology,
  StateResponse,
  StreamDoneEvent,
  TraceExecutionMode,
  TraceFidelity,
  TokenStepCapture,
} from "@/lib/types";

const DEFAULT_PROMPT =
  "Trace the attention route you would use to explain why masking a single head can change a model's response.";

const TRACE_MODEL_OPTIONS = [
  {
    value: "gpt2",
    label: "GPT-2",
    detail: "12 layers x 12 heads = 144 total heads",
  },
  {
    value: "sshleifer/tiny-gpt2",
    label: "Tiny GPT-2",
    detail: "2 layers x 2 heads = 4 total heads",
  },
];

const EXECUTION_MODE_OPTIONS: Array<{
  value: TraceExecutionMode;
  label: string;
  description: string;
}> = [
  {
    value: "auto",
    label: "Auto",
    description: "Prefer Ollama when it is available, otherwise fall back to exact inline tracing.",
  },
  {
    value: "faithful",
    label: "Faithful",
    description: "Generate inside the traced Hugging Face model for exact causal evidence.",
  },
];

export function SynapseDashboard() {
  const [state, setState] = useState<StateResponse | null>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [systemPrompt, setSystemPrompt] = useState(
    "You are the instrumented local model inside Synapse-Graph. Explain your reasoning concisely.",
  );
  const [responseText, setResponseText] = useState("");
  const [trace, setTrace] = useState<AttentionTrace | null>(null);
  const [selectedLayerIndex, setSelectedLayerIndex] = useState(0);
  const [discovery, setDiscovery] = useState<import("@/lib/types").CircuitDiscoveryResponse | null>(null);
  const [overlayTrace, setOverlayTrace] = useState<import("@/lib/types").AttentionTrace | null>(null);
  const [targetToken, setTargetToken] = useState<string>("");
  const [executionMode, setExecutionMode] = useState<TraceExecutionMode>("faithful");
  const [maxNewTokens, setMaxNewTokens] = useState(32);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(0.95);
  const [traceModelName, setTraceModelName] = useState("gpt2");
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isQuarantining, setIsQuarantining] = useState(false);
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [discoverIdx, setDiscoverIdx] = useState<number>(0);
  const [discoverTotal, setDiscoverTotal] = useState<number>(0);
  const [quarantineIdx, setQuarantineIdx] = useState<number>(0);
  const [quarantineTotal, setQuarantineTotal] = useState<number>(0);
  const [discoverTopK, setDiscoverTopK] = useState<number>(3);
  const [discoverPairSweeps, setDiscoverPairSweeps] = useState<number>(3);
  const [isSyncingDefects, setIsSyncingDefects] = useState(false);
  const [systemEditable, setSystemEditable] = useState(false);
  const [systemChatVisible, setSystemChatVisible] = useState(false);
  const [systemChatInput, setSystemChatInput] = useState("");

  function appendLog(channel: string, message: string, detail?: string) {
    setLogs((current) => {
      const nextEntry: LogEntry = {
        id: crypto.randomUUID(),
        channel,
        message,
        detail,
        createdAt: new Date().toLocaleTimeString(),
      };

      return [nextEntry, ...current].slice(0, 48);
    });
  }

  const deferredTrace = useDeferredValue(trace);
  const topology = state?.topology ?? null;
  const maskedHeads = mergeHeadMasks(
    state?.masked_heads ?? [],
    state?.openmetadata.defective_heads ?? [],
  );
  const latestStep = deferredTrace?.steps.at(-1) ?? null;
  const selectedLayer = latestStep?.layers.find((layer) => layer.layer_index === selectedLayerIndex) ?? null;
  const activeTrace = deferredTrace ?? trace;
  const predictedBackend = resolvePredictedBackend(executionMode, state?.ollama_available ?? false);
  const predictedFidelity = resolvePredictedFidelity(executionMode, state?.ollama_available ?? false);
  const currentBackend = activeTrace?.generation_backend ?? null;
  const currentFidelity = activeTrace?.trace_fidelity ?? null;

  useEffect(() => {
    void loadInitialState();
  }, []);

  async function loadInitialState() {
    try {
      const nextState = await fetchState();
      setState(nextState);
      setResponseText(nextState.latest_session?.response_text ?? "");
      setTrace(nextState.latest_session?.trace ?? null);
      const lastLayer = nextState.latest_session?.trace.steps.at(-1)?.layers.at(-1);
      setSelectedLayerIndex(lastLayer?.layer_index ?? 0);
      appendLog("BOOT", "Hydrated dashboard state from the neural proxy.");
      // If the topology is empty, attempt to trigger the HF preload endpoint
      // so the shadow tracer can load a fallback model and populate topology.
      if ((nextState.topology?.total_layers ?? 0) === 0) {
        appendLog("BOOT", "Topology empty — requesting HF preload to populate tracer.");
        try {
          const preloaded = await preloadHF();
          setState(preloaded);
          setResponseText(preloaded.latest_session?.response_text ?? "");
          setTrace(preloaded.latest_session?.trace ?? null);
          appendLog("BOOT", "HF preload completed; topology refreshed.");
        } catch (err) {
          appendLog("ERROR", "HF preload request failed.", err instanceof Error ? err.message : String(err));
        }
      }
    } catch (loadError) {
      const message =
        loadError instanceof Error ? loadError.message : "Failed to fetch initial state.";
      setError(message);
      appendLog("ERROR", "State bootstrap failed.", message);
    }
  }

  async function handleSyncDefects() {
    setIsSyncingDefects(true);
    try {
      // Perform canonical sync then refresh the full runtime state to ensure UI consistency
      await syncOpenMetadataDefects();
      const refreshed = await fetchState();
      setState(refreshed);
      appendLog("SYNC", `Synchronized ${refreshed.masked_heads.length} defective heads from OpenMetadata.`);
    } catch (syncError) {
      const message = syncError instanceof Error ? syncError.message : "Defect sync failed.";
      setError(message);
      appendLog("ERROR", "OpenMetadata defect sync failed.", message);
    } finally {
      setIsSyncingDefects(false);
    }
  }

  async function handleProbe(execMode?: TraceExecutionMode) {
    const runMode = execMode ?? executionMode;
    setIsRunning(true);
    setError(null);
    setResponseText("");
    setTrace(null);
    appendLog("RUN", "Dispatching a new generation probe.");

    try {
      await streamGeneration(
        {
          prompt,
          system_prompt: systemPrompt,
          max_new_tokens: maxNewTokens,
          temperature,
          top_p: topP,
          stop: [],
          stream: true,
          execution_mode: runMode,
          trace_model_name: traceModelName,
        },
        {
          onSession: (event) => {
            startTransition(() => {
              setState((current) => ({
                ...(current ?? emptyState(event.topology)),
                topology: event.topology,
                masked_heads: event.maskedHeads,
                openmetadata: event.openmetadata,
              }));
            });
            appendLog(
              "SESSION",
              `Session ${event.sessionId.slice(0, 8)} opened with ${event.topology.total_layers} traced layers in ${executionModeLabel(runMode)} mode.`,
            );
          },
          onToken: (event) => {
            setResponseText((current) => current + event.token);
          },
          onTraceStep: (event) => {
            startTransition(() => {
              setTrace((current) =>
                upsertTrace(
                  current,
                  event.step,
                  prompt,
                  topology,
                  runMode,
                  state?.ollama_available ?? false,
                ),
              );
            });
            setSelectedLayerIndex(event.step.layers.at(-1)?.layer_index ?? 0);
            appendLog(
              "TRACE",
              `Step ${event.step.step_index} activated ${event.step.high_activation_path.length} lineage edges.`,
              event.step.high_activation_path.join(" -> "),
            );
          },
          onDone: (event) => {
            handleDoneEvent(event);
          },
          onError: (event) => {
            setError(event.message);
            appendLog("ERROR", "Streaming generation failed.", event.message);
          },
        },
      );

      const refreshedState = await fetchState();
      setState(refreshedState);
    } catch (streamError) {
      const message = streamError instanceof Error ? streamError.message : "Probe failed.";
      setError(message);
      appendLog("ERROR", "Probe request failed before streaming completed.", message);
    } finally {
      setIsRunning(false);
    }
  }

  async function handleDiscoverCircuit() {
    console.debug("handleDiscoverCircuit invoked");
    setError(null);
    appendLog("DISCOVERY", "Starting circuit discovery sweep.");
    setIsDiscovering(true);
    const req: import("@/lib/types").CircuitDiscoveryRequest = {
      prompt,
      target_hallucination_token: targetToken || (prompt.split(" ").at(-1) ?? ""),
      trace_model_name: traceModelName,
      max_new_tokens: maxNewTokens,
      // Use interactive values from UI controls
      top_k_heads: discoverTopK,
      max_pair_sweeps: discoverPairSweeps,
    };

    try {
      await streamDiscoverCircuit(req, {
        onProgress: (data) => {
          appendLog("DISCOVERY", data?.message ?? JSON.stringify(data));
          if (data?.message === "running_sweeps" && typeof data.count === "number") {
            setDiscoverTotal(data.count);
            setDiscoverIdx(0);
          }
          if (data?.message === "baseline_generation_done") {
            setDiscoverIdx(0);
          }
        },
        onSweepStart: (data) => {
          setDiscoverIdx(typeof data.idx === "number" ? data.idx : 0);
          setDiscoverTotal(typeof data.total === "number" ? data.total : 0);
          appendLog("DISCOVERY", `Sweep ${data.idx}/${data.total} starting: ${JSON.stringify(data.masks)}`);
        },
        onSweepDone: (data) => {
          setDiscoverIdx(typeof data.idx === "number" ? data.idx : 0);
          appendLog("DISCOVERY", `Sweep ${data.idx} done: target_count=${data.target_count} duration=${data.duration?.toFixed?.(3) ?? data.duration}`);
        },
        onDone: (data) => {
          try {
            const payload = typeof data.response === "string" ? JSON.parse(data.response) : data.response;
            setDiscovery(payload as import("@/lib/types").CircuitDiscoveryResponse);
            setOverlayTrace(null);
            appendLog("DISCOVERY", `Discovery completed with ${payload.sweep_results?.length ?? 0} sweeps.`);
          } catch (e) {
            appendLog("ERROR", "Failed to parse discovery result.", String(e));
          }
          setIsDiscovering(false);
          setDiscoverIdx(0);
          setDiscoverTotal(0);
        },
        onError: (err) => {
          appendLog("ERROR", "Discovery stream error.", err?.message ?? JSON.stringify(err));
          setError(err?.message ?? "Discovery failed.");
          setIsDiscovering(false);
          setDiscoverIdx(0);
          setDiscoverTotal(0);
        },
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      appendLog("ERROR", "Discovery failed.", message);
      setIsDiscovering(false);
    }
  }

  async function handleViewSweepResult(index: number) {
    if (!discovery) return;
    console.debug(`handleViewSweepResult invoked for index=${index}`);
    const sweep = discovery.sweep_results[index];
    setOverlayTrace(sweep.trace ?? null);
    appendLog("DISCOVERY", `Showing overlay for sweep #${index + 1} (effect ${sweep.causal_effect_score}).`);
  }

  async function handleQuarantineSweepResult(index: number) {
    if (!discovery) return;
    console.debug(`handleQuarantineSweepResult invoked for index=${index}`);
    const sweep = discovery.sweep_results[index];
    try {
      const payload: import("@/lib/types").QuarantineRequest = {
        heads: sweep.masked_heads,
        reason: `sweep_${index + 1}`,
      };

      setIsQuarantining(true);
      appendLog("QUARANTINE", `Starting quarantine for sweep #${index + 1}`);
      await streamQuarantineCircuit(payload, {
        onProgress: (data) => {
          appendLog("QUARANTINE", data?.message ?? JSON.stringify(data));
          if (data?.message === "quarantine_start" && typeof data.total === "number") {
            setQuarantineTotal(data.total);
            setQuarantineIdx(0);
          }
          if ((data?.message === "quarantine_pair_start" || data?.message === "quarantine_pair_done") && typeof data.idx === "number") {
            setQuarantineIdx(data.idx);
            if (typeof data.total === "number") setQuarantineTotal(data.total);
          }
        },
        onDone: (data) => {
          appendLog("QUARANTINE", data?.reason ?? "Quarantine completed");
          // Attempt to apply server-provided masked heads into UI immediately
          try {
            const parsed = {
              parsed_heads: Array.isArray(data.parsed_heads) ? data.parsed_heads.map((p: any) => (typeof p === "string" ? JSON.parse(p) : p)) : [],
              masked_heads: Array.isArray(data.masked_heads) ? data.masked_heads.map((m: any) => (typeof m === "string" ? JSON.parse(m) : m)) : [],
            };
            setState((current) => {
              const currTopology = current?.topology ?? null;
              const nextState = current ?? emptyState(currTopology);
              return {
                ...nextState,
                masked_heads: parsed.masked_heads ?? nextState.masked_heads,
                openmetadata: {
                  ...nextState.openmetadata,
                  defective_heads: parsed.parsed_heads ?? nextState.openmetadata.defective_heads,
                  connected: true,
                },
              };
            });
          } catch (e) {
            console.debug("Failed to apply quarantine done payload to UI", e);
          }

          // Try canonical refresh
          void (async () => {
            try {
              const refreshed = await fetchState();
              setState(refreshed);
            } catch (e) {
              console.debug("fetchState after stream quarantine failed", e);
            }
          })();

          setTimeout(() => {
            document.getElementById("governance-panel")?.scrollIntoView({ behavior: "smooth", block: "center" });
          }, 80);
          setIsQuarantining(false);
        },
        onError: (err) => {
          appendLog("ERROR", "Quarantine stream error.", err?.message ?? JSON.stringify(err));
          setIsQuarantining(false);
          setQuarantineIdx(0);
          setQuarantineTotal(0);
        },
      });
    } catch (err) {
      appendLog("ERROR", "Quarantine sweep failed.", err instanceof Error ? err.message : String(err));
      setIsQuarantining(false);
    }
  }

  async function handleQuarantineDiscoveredCircuit() {
    console.debug("handleQuarantineDiscoveredCircuit invoked");
    if (!discovery) return;
    setIsQuarantining(true);
    appendLog("QUARANTINE", "Dispatching quarantine to OpenMetadata...");
    const payload: import("@/lib/types").QuarantineRequest = {
      heads: discovery.discovered_circuit,
      reason: "discovered_via_ui",
    };

    try {
      await streamQuarantineCircuit(payload, {
        onProgress: (data) => {
          appendLog("QUARANTINE", data?.message ?? JSON.stringify(data));
          if (data?.message === "quarantine_start" && typeof data.total === "number") {
            setQuarantineTotal(data.total);
            setQuarantineIdx(0);
          }
          if ((data?.message === "quarantine_pair_start" || data?.message === "quarantine_pair_done") && typeof data.idx === "number") {
            setQuarantineIdx(data.idx);
            if (typeof data.total === "number") setQuarantineTotal(data.total);
          }
        },
        onDone: (data) => {
          appendLog("QUARANTINE", data?.reason ?? "Quarantine completed");
          try {
            const parsed = {
              parsed_heads: Array.isArray(data.parsed_heads) ? data.parsed_heads.map((p: any) => (typeof p === "string" ? JSON.parse(p) : p)) : [],
              masked_heads: Array.isArray(data.masked_heads) ? data.masked_heads.map((m: any) => (typeof m === "string" ? JSON.parse(m) : m)) : [],
            };
            setState((current) => {
              const currTopology = current?.topology ?? null;
              const nextState = current ?? emptyState(currTopology);
              return {
                ...nextState,
                masked_heads: parsed.masked_heads ?? nextState.masked_heads,
                openmetadata: {
                  ...nextState.openmetadata,
                  defective_heads: parsed.parsed_heads ?? nextState.openmetadata.defective_heads,
                  connected: true,
                },
              };
            });
          } catch (e) {
            console.debug("Failed to apply quarantine done payload to UI", e);
          }

          void (async () => {
            try {
              const refreshed = await fetchState();
              setState(refreshed);
            } catch (e) {
              console.debug("fetchState after stream quarantine failed", e);
            }
          })();

          setTimeout(() => {
            document.getElementById("governance-panel")?.scrollIntoView({ behavior: "smooth", block: "center" });
          }, 80);
          setIsQuarantining(false);
          setQuarantineIdx(0);
          setQuarantineTotal(0);
        },
        onError: (err) => {
          appendLog("ERROR", "Quarantine stream error.", err?.message ?? JSON.stringify(err));
          setIsQuarantining(false);
          setQuarantineIdx(0);
          setQuarantineTotal(0);
        },
      });
    } catch (qErr) {
      appendLog("ERROR", "Quarantine failed.", qErr instanceof Error ? qErr.message : String(qErr));
      setIsQuarantining(false);
    }
  }

  function handleDoneEvent(event: StreamDoneEvent) {
    setResponseText(event.responseText);
    setTrace(event.trace);
    setState((current) => {
      const currentTopology = current?.topology ?? null;
      const nextState = current ?? emptyState(currentTopology);
      return {
        ...nextState,
        masked_heads: event.maskedHeads,
      };
    });
    setSelectedLayerIndex(event.trace.steps.at(-1)?.layers.at(-1)?.layer_index ?? 0);
    appendLog("DONE", `Generation session ${event.sessionId.slice(0, 8)} completed.`);
  }

  function applyExactTracePreset() {
    setExecutionMode("faithful");
    setMaxNewTokens(32);
    setTemperature(0);
    setTopP(0.95);
    setTraceModelName("gpt2");
  }

  function applyReadableAnswerPreset() {
    setExecutionMode("auto");
    setMaxNewTokens(160);
    setTemperature(0.2);
    setTopP(0.95);
  }

  return (
    <main className="min-h-screen px-4 py-6 sm:px-6 xl:px-8">
      <div className="mx-auto max-w-[1680px] space-y-5">
        <header className="panel-shell rounded-sm px-5 py-4">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="panel-label">Synapse-Graph / The LLM Glassbox</p>
              <h1 className="mt-3 text-3xl font-semibold tracking-tight text-zinc-50">
                Neural lineage, governance, and live head quarantine in one control room.
              </h1>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-muted">
                Switch between fast shadow tracing and faithful inline tracing so the same dashboard
                can show either low-latency operator telemetry or exact token-level evidence.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                icon={Orbit}
                label="Generation Backend"
                value={currentBackend ? backendLabel(currentBackend) : "Awaiting run"}
                detail={currentBackend ? executionModeLabel(executionMode) : `Planned: ${backendLabel(predictedBackend)}`}
              />
              <MetricCard
                icon={Radio}
                label="Trace Fidelity"
                value={currentFidelity ? fidelityLabel(currentFidelity) : "Not verified"}
                detail={activeTrace?.analysis_mode ?? `Planned: ${predictedAnalysisMode(executionMode, state?.ollama_available ?? false)}`}
              />
              <MetricCard
                icon={ShieldAlert}
                label="Lineage Depth"
                value={`${latestStep?.high_activation_path.length ?? 0} active hops`}
                detail={
                  latestStep?.explanation
                    ? truncateText(latestStep.explanation, 96)
                    : "No step evidence yet"
                }
              />
              <MetricCard
                icon={Sparkles}
                label="Masked Heads"
                value={`${maskedHeads.length}`}
                detail={maskedHeads.length ? "Applied to next trace" : state?.openmetadata.connected ? "OM synchronized" : "Local only"}
              />
            </div>
          </div>
        </header>

        <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)_360px]">
          <section className="space-y-5">
            <div className="panel-shell rounded-sm p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="panel-label">Probe Console</p>
                  <h2 className="mt-2 text-lg font-medium text-zinc-50">Interrogate the model</h2>
                </div>
                <button
                  type="button"
                  onClick={handleSyncDefects}
                  disabled={isSyncingDefects}
                  className={`border border-line bg-panel2 px-3 py-2 text-xs uppercase tracking-[0.22em] transition ${
                    isSyncingDefects ? "text-muted cursor-not-allowed" : "text-zinc-200 hover:border-accent hover:text-accent"
                  }`}
                >
                  {isSyncingDefects ? "Syncing..." : "Sync Defects"}
                </button>
              </div>

              <div className="mt-5 space-y-4">
                <div className="grid gap-2 sm:grid-cols-2">
                  <button
                    type="button"
                    onClick={applyExactTracePreset}
                    className="rounded-sm border border-accent/35 bg-accent/8 px-3 py-3 text-left transition hover:border-accent"
                  >
                    <span className="metric-mono text-xs uppercase tracking-[0.22em] text-accent">
                      Exact Trace
                    </span>
                    <p className="mt-2 text-xs leading-5 text-muted">
                      Short deterministic HF run for real layer/head evidence.
                    </p>
                  </button>
                  <button
                    type="button"
                    onClick={applyReadableAnswerPreset}
                    className="rounded-sm border border-line bg-panel2 px-3 py-3 text-left transition hover:border-accent/45"
                  >
                    <span className="metric-mono text-xs uppercase tracking-[0.22em] text-zinc-200">
                      Readable Answer
                    </span>
                    <p className="mt-2 text-xs leading-5 text-muted">
                      Longer Ollama response with proxy or shadow evidence.
                    </p>
                  </button>
                </div>

                <div className="border border-line bg-panel2/60 p-3">
                  <label className="block">
                    <span className="panel-label">Trace Model</span>
                    <select
                      value={traceModelName}
                      onChange={(event) => setTraceModelName(event.target.value)}
                      className="mt-2 w-full rounded-sm border border-line bg-panel px-3 py-2 text-sm text-zinc-100 outline-none focus:border-accent"
                    >
                      {TRACE_MODEL_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label} - {option.detail}
                        </option>
                      ))}
                    </select>
                    <p className="mt-2 text-xs leading-5 text-muted">
                      This controls the real Hugging Face model used for layer/head tracing. More
                      heads means a larger graph and slower exact runs.
                    </p>
                  </label>

                  <div className="mt-3 grid gap-3 sm:grid-cols-3">
                    <label className="block">
                      <span className="panel-label">Tokens</span>
                      <input
                        type="number"
                        min={1}
                        max={2048}
                        value={maxNewTokens}
                        onChange={(event) =>
                          setMaxNewTokens(clampNumber(event.target.valueAsNumber, 1, 2048))
                        }
                        className="mt-2 w-full rounded-sm border border-line bg-panel px-3 py-2 text-sm text-zinc-100 outline-none focus:border-accent"
                      />
                    </label>
                    <label className="block">
                      <span className="panel-label">Temp</span>
                      <input
                        type="number"
                        min={0}
                        max={2}
                        step={0.1}
                        value={temperature}
                        onChange={(event) =>
                          setTemperature(clampNumber(event.target.valueAsNumber, 0, 2))
                        }
                        className="mt-2 w-full rounded-sm border border-line bg-panel px-3 py-2 text-sm text-zinc-100 outline-none focus:border-accent"
                      />
                    </label>
                    <label className="block">
                      <span className="panel-label">Top P</span>
                      <input
                        type="number"
                        min={0.01}
                        max={1}
                        step={0.01}
                        value={topP}
                        onChange={(event) =>
                          setTopP(clampNumber(event.target.valueAsNumber, 0.01, 1))
                        }
                        className="mt-2 w-full rounded-sm border border-line bg-panel px-3 py-2 text-sm text-zinc-100 outline-none focus:border-accent"
                      />
                    </label>
                  </div>
                  <p className="mt-3 text-xs leading-5 text-muted">
                    {executionMode === "faithful"
                      ? "Faithful mode traces the Hugging Face model exactly. Tiny tracer models are useful for evidence, not polished prose."
                      : "Auto mode favors Ollama for better prose. Treat the trace as proxy unless a matching HF shadow trace is available."}
                  </p>
                </div>

                <label className="block">
                  <div className="flex items-center justify-between">
                    <span className="panel-label">System Prompt</span>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => setSystemChatVisible((v) => !v)}
                        className="rounded-sm border border-line bg-panel px-2 py-1 text-xs text-zinc-200 hover:border-accent"
                      >
                        {systemChatVisible ? "Hide Chat" : "Chat"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setSystemEditable((e) => !e)}
                        className="rounded-sm border border-accent/30 bg-accent/8 px-2 py-1 text-xs text-accent"
                      >
                        {systemEditable ? "Save" : "Edit"}
                      </button>
                    </div>
                  </div>

                  {systemEditable ? (
                    <textarea
                      value={systemPrompt}
                      onChange={(event) => setSystemPrompt(event.target.value)}
                      className="mt-2 h-28 w-full resize-none rounded-sm border border-line bg-panel2 px-3 py-3 text-sm text-zinc-100 outline-none transition focus:border-accent"
                    />
                  ) : (
                    <div className="mt-2 rounded-sm border border-line bg-panel2 p-3 text-sm text-zinc-100 whitespace-pre-wrap">
                      {systemPrompt}
                    </div>
                  )}

                  <p className="mt-2 text-xs leading-5 text-muted">
                    Sets the model role/instructions. Instruction-tuned models follow this better than
                    base GPT-2 models. Editing is disabled by default; click Edit to modify.
                  </p>

                  {systemChatVisible ? (
                    <div className="mt-3 grid grid-cols-[1fr_auto] gap-2">
                      <input
                        type="text"
                        placeholder="Ask the system (mini-chatbot)"
                        className="rounded-sm border border-line bg-panel px-3 py-2 text-sm text-zinc-100 outline-none"
                        value={systemChatInput}
                        onChange={(e) => setSystemChatInput(e.target.value)}
                      />
                      <button
                        type="button"
                        onClick={async () => {
                          if (!systemChatInput) return;
                          // Send a single-shot generation using the system prompt as instruction
                          setIsRunning(true);
                          setResponseText("");
                          setTrace(null);
                          appendLog("SYSCHAT", `System chat: ${systemChatInput.slice(0, 120)}`);
                          try {
                            await streamGeneration(
                              {
                                prompt: systemChatInput,
                                system_prompt: systemPrompt,
                                max_new_tokens: 80,
                                temperature: 0.2,
                                top_p: 0.95,
                                stop: [],
                                stream: true,
                                execution_mode: "auto",
                                trace_model_name: traceModelName,
                              },
                              {
                                onSession: (event) => {
                                  startTransition(() => {
                                    setState((current) => ({
                                      ...(current ?? emptyState(event.topology)),
                                      topology: event.topology,
                                      masked_heads: event.maskedHeads,
                                      openmetadata: event.openmetadata,
                                    }));
                                  });
                                },
                                onToken: (event) => {
                                  setResponseText((current) => current + event.token);
                                },
                                onTraceStep: (event) => {
                                  startTransition(() => {
                                    setTrace((current) => upsertTrace(current, event.step, systemChatInput, state?.topology ?? null, "auto", state?.ollama_available ?? false));
                                  });
                                },
                                onDone: (event) => {
                                  handleDoneEvent(event);
                                },
                                onError: (event) => {
                                  setError(event.message);
                                  appendLog("ERROR", "System chat failed.", event.message);
                                },
                              },
                            );

                            const refreshed = await fetchState();
                            setState(refreshed);
                          } catch (err: any) {
                            const message = err instanceof Error ? err.message : String(err);
                            setError(message);
                            appendLog("ERROR", "System chat failed.", message);
                          } finally {
                            setIsRunning(false);
                            setSystemChatInput("");
                          }
                        }}
                        disabled={isRunning || !systemChatInput}
                        className="rounded-sm border border-accent bg-accent/12 px-3 py-2 text-xs text-accent disabled:cursor-not-allowed disabled:border-line disabled:bg-panel2 disabled:text-muted"
                      >
                        Ask
                      </button>
                    </div>
                  ) : null}
                </label>

                <label className="block">
                  <span className="panel-label">User Prompt</span>
                  <textarea
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    className="mt-2 h-40 w-full resize-none rounded-sm border border-line bg-panel2 px-3 py-3 text-sm text-zinc-100 outline-none transition focus:border-accent"
                  />
                  <p className="mt-2 text-xs leading-5 text-muted">
                    This is the actual input being traced. In Faithful mode, the graph highlights
                    which layers and heads attended while continuing this text.
                  </p>
                </label>

                <div>
                  <span className="panel-label">Execution Mode</span>
                  <div className="mt-2 grid gap-2">
                    {EXECUTION_MODE_OPTIONS.map((option) => {
                      const isSelected = executionMode === option.value;

                      return (
                        <button
                          key={option.value}
                          type="button"
                          onClick={() => setExecutionMode(option.value)}
                          className={`rounded-sm border px-3 py-3 text-left transition ${
                            isSelected
                              ? "border-accent bg-accent/10 text-zinc-50"
                              : "border-line bg-panel2 text-zinc-200 hover:border-accent/45"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <span className="metric-mono text-xs uppercase tracking-[0.22em]">
                              {option.label}
                            </span>
                            <span className="text-[11px] text-muted">
                              {modeRuntimeHint(option.value, state?.ollama_available ?? false)}
                            </span>
                          </div>
                          <p className="mt-2 text-xs leading-5 text-muted">{option.description}</p>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => handleProbe()}
                    disabled={isRunning}
                    className="accent-glow border border-accent bg-accent/12 px-4 py-3 text-xs font-medium uppercase tracking-[0.24em] text-accent transition hover:bg-accent/18 disabled:cursor-not-allowed disabled:border-line disabled:bg-panel2 disabled:text-muted"
                  >
                    {isRunning ? "Streaming" : "Probe Model"}
                  </button>
                  <div className="metric-mono text-xs text-muted">
                    <p>{topology?.model_name ?? "Topology not loaded"}</p>
                    <p>{state?.openmetadata.catalog_ready ? "Metadata graph armed" : "Metadata graph pending"}</p>
                  </div>
                </div>
              </div>
            </div>

            <ConsoleLog logs={logs} />
          </section>

          <section>
            <SynapseGraph
              topology={topology}
                trace={deferredTrace}
                overlayTrace={overlayTrace}
              maskedHeads={maskedHeads}
              onSelectLayer={setSelectedLayerIndex}
            />
          </section>

          <section className="space-y-5">
            <ActivationChart layer={selectedLayer} />

            <ExplainabilityPanel trace={activeTrace} latestStep={latestStep} onRerunFaithful={() => handleProbe("faithful")} isRunning={isRunning} />
            <div className="panel-shell rounded-sm p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="panel-label">Causal Discovery</p>
                  <h3 className="mt-2 text-lg font-medium text-zinc-50">Sweep heads to find causal circuits</h3>
                </div>
                <div className="metric-mono text-right text-xs text-muted">Discovery: quick</div>
              </div>

              <div className="mt-3 space-y-3">
                <p className="text-xs text-muted">Target token: the token you consider a hallucination to remove.</p>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <input
                    type="text"
                    placeholder="target token (e.g. green)"
                    className="rounded-sm border border-line bg-panel px-3 py-2 text-sm text-zinc-100 outline-none"
                    value={targetToken}
                    onChange={(e) => setTargetToken(e.target.value)}
                  />
                  <button
                    type="button"
                    onClick={() => void handleDiscoverCircuit()}
                    disabled={isDiscovering}
                    className="rounded-sm border border-accent bg-accent/12 px-3 py-2 text-xs text-accent disabled:cursor-not-allowed disabled:border-line disabled:bg-panel2 disabled:text-muted"
                  >
                    {isDiscovering
                      ? discoverTotal > 0
                        ? `Discovering ${discoverIdx}/${discoverTotal}`
                        : "Discovering..."
                      : "Run Discovery"}
                  </button>
                </div>

                <div className="mt-2 grid grid-cols-2 gap-2">
                  <label className="block">
                    <span className="panel-label">Top K</span>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      placeholder="top_k_heads (number of top heads to test)"
                      className="mt-2 w-full rounded-sm border border-line bg-panel px-3 py-2 text-sm text-zinc-100 outline-none"
                      value={discoverTopK}
                      onChange={(e) => setDiscoverTopK(clampNumber(Number(e.target.value || 0), 1, 20))}
                    />
                    <p className="mt-1 text-xs text-muted">Number of top heads to sweep for causal effect.</p>
                  </label>

                  <label className="block">
                    <span className="panel-label">Pair Sweeps</span>
                    <input
                      type="number"
                      min={0}
                      max={190}
                      placeholder="max_pair_sweeps (how many pair combinations to try)"
                      className="mt-2 w-full rounded-sm border border-line bg-panel px-3 py-2 text-sm text-zinc-100 outline-none"
                      value={discoverPairSweeps}
                      onChange={(e) => setDiscoverPairSweeps(clampNumber(Number(e.target.value || 0), 0, 190))}
                    />
                    <p className="mt-1 text-xs text-muted">How many head-pair combinations to attempt.</p>
                  </label>
                </div>

                {discovery ? (
                  <div className="mt-2 space-y-2">
                    <div className="border border-line bg-panel/70 p-2">
                      <p className="text-sm text-zinc-100">Discovered Circuit</p>
                      <p className="mt-1 text-xs text-muted">Combined effect: {discovery.combined_causal_effect}</p>
                      <div className="mt-2 flex gap-2 flex-wrap">
                        {discovery.discovered_circuit.map((h) => (
                          <span key={`${h.layer_index}-${h.head_index}`} className="metric-mono border border-accent/25 bg-accent/8 px-2 py-1 text-[11px] text-accent">
                            {h.layer_name}:{h.head_name}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="thin-scrollbar max-h-36 overflow-y-auto space-y-2">
                      {discovery.sweep_results.map((s, i) => (
                        <div key={i} className="border border-line p-2">
                          <div className="flex items-center justify-between gap-2">
                            <div>
                              <p className="text-xs text-zinc-100">Sweep #{i + 1}</p>
                              <p className="mt-1 text-xs text-muted">Effect: {s.causal_effect_score} • Similarity: {s.text_similarity}</p>
                            </div>
                            <div className="flex gap-2">
                              <button className="rounded-sm border border-line px-2 py-1 text-xs" onClick={() => void handleViewSweepResult(i)}>View Overlay</button>
                              <button
                                className="rounded-sm border border-rose-500 px-2 py-1 text-xs text-rose-300"
                                onClick={() => void handleQuarantineSweepResult(i)}
                                disabled={isQuarantining}
                              >
                                {isQuarantining
                                  ? quarantineTotal > 0
                                    ? `Quarantining ${quarantineIdx}/${quarantineTotal}`
                                    : "Quarantining..."
                                  : "Quarantine Sweep"}
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="mt-2 flex gap-2">
                              <button
                                type="button"
                                onClick={() => void handleQuarantineDiscoveredCircuit()}
                                disabled={isQuarantining}
                                className="rounded-sm border border-rose-500 bg-rose-500/10 px-3 py-2 text-xs text-rose-300 disabled:cursor-not-allowed disabled:border-line disabled:bg-panel2 disabled:text-muted"
                              >
                                {isQuarantining
                                  ? quarantineTotal > 0
                                    ? `Quarantining ${quarantineIdx}/${quarantineTotal}`
                                    : "Quarantining..."
                                  : "QUARANTINE CIRCUIT IN OPENMETADATA"}
                              </button>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

            <ResponsePanel responseText={responseText} latestStep={latestStep} error={error} />

            <GovernancePanel
              topology={topology}
              maskedHeads={maskedHeads}
              state={state}
              selectedLayer={selectedLayer}
              onStateChange={setState}
              appendLog={appendLog}
              syncDefects={handleSyncDefects}
              isSyncingDefects={isSyncingDefects}
            />
          </section>
        </div>
      </div>
    </main>
  );

}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Orbit;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="panel-shell min-w-[180px] rounded-sm px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <p className="panel-label">{label}</p>
        <Icon className="h-4 w-4 text-accent" strokeWidth={1.5} />
      </div>
      <p className="metric-mono mt-3 text-lg text-zinc-50">{value}</p>
      <p className="mt-1 text-xs text-muted">{detail}</p>
    </div>
  );
}

function ResponsePanel({
  responseText,
  latestStep,
  error,
}: {
  responseText: string;
  latestStep: TokenStepCapture | null;
  error: string | null;
}) {
  return (
    <div id="governance-panel" className="panel-shell rounded-sm p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="panel-label">Response Stream</p>
          <h3 className="mt-2 text-lg font-medium text-zinc-50">Generation output</h3>
        </div>
        <div className="metric-mono text-right text-xs text-muted">
          <p>{latestStep ? `step ${latestStep.step_index + 1}` : "No trace"}</p>
          <p>{latestStep ? `${latestStep.prompt_plus_generation_length} total tokens` : "Awaiting run"}</p>
        </div>
      </div>
      <div className="thin-scrollbar mt-4 h-64 overflow-y-auto border border-line bg-panel2/80 p-4 text-sm leading-7 text-zinc-100">
        {responseText || "No response yet. Fire a probe to stream tokens here."}
      </div>
      {error ? <p className="mt-3 text-sm text-rose-400">{error}</p> : null}
    </div>
  );
}

function ExplainabilityPanel({
  trace,
  latestStep,
  onRerunFaithful,
  isRunning,
}: {
  trace: AttentionTrace | null;
  latestStep: TokenStepCapture | null;
  onRerunFaithful?: () => void;
  isRunning?: boolean;
}) {
  const summary = trace?.summary ?? null;
  const evidenceQuality = trace?.evidence_quality ?? null;

  return (
    <div className="panel-shell rounded-sm p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="panel-label">Glassbox Summary</p>
          <h3 className="mt-2 text-lg font-medium text-zinc-50">Why the model responded this way</h3>
        </div>
        <div className="metric-mono text-right text-xs text-muted">
          <p>{trace ? fidelityLabel(trace.trace_fidelity) : "Awaiting run"}</p>
          <p>{trace?.analysis_mode ?? "No analysis mode yet"}</p>
          {trace?.match_score != null ? (
            <p>{`Match ${(trace.match_score * 100).toFixed(1)}%`}</p>
          ) : null}
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {trace?.trace_fidelity === "proxy" ? (
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => onRerunFaithful?.()}
              disabled={isRunning}
              className="rounded-sm border border-accent/25 bg-panel2 px-2 py-1 text-[11px] text-accent"
            >
              Re-run Faithful
            </button>
          </div>
        ) : null}
        <div className="border border-line bg-panel2/70 p-3">
          <p className="text-sm leading-6 text-zinc-100">
            {summary?.explanation ??
              latestStep?.explanation ??
              "Run a probe to see the dominant layers, heads, and source tokens behind the output."}
          </p>
        </div>

        {evidenceQuality ? (
          <div className="border border-line bg-panel2/60 p-3">
            <div className="flex items-center justify-between gap-3">
              <p className="panel-label">Evidence Quality</p>
              <p className="metric-mono text-xs text-zinc-100">
                {evidenceQuality.label.toUpperCase()} / {(evidenceQuality.score * 100).toFixed(0)}%
              </p>
            </div>
            <p className="mt-2 text-xs leading-5 text-muted">{evidenceQuality.exactness}</p>
            <p className="mt-2 metric-mono text-[11px] uppercase tracking-[0.2em] text-muted">
              {evidenceQuality.causal_validation.replaceAll("_", " ")}
            </p>
            {evidenceQuality.black_box_gaps.length ? (
              <div className="mt-3 space-y-1">
                {evidenceQuality.black_box_gaps.slice(0, 3).map((gap) => (
                  <p key={gap} className="text-xs leading-5 text-amber-200">
                    {gap}
                  </p>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        <ChipGroup
          label="Dominant Heads"
          items={summary?.dominant_heads ?? latestStep?.high_activation_path ?? []}
          emptyLabel="No dominant heads captured yet."
        />

        <ChipGroup
          label="Influential Tokens"
          items={summary?.influential_tokens ?? latestStep?.evidence_tokens ?? []}
          emptyLabel="No evidence tokens captured yet."
        />
      </div>
    </div>
  );
}

function ChipGroup({
  label,
  items,
  emptyLabel,
}: {
  label: string;
  items: string[];
  emptyLabel: string;
}) {
  return (
    <div className="border border-line bg-panel2/60 p-3">
      <p className="panel-label">{label}</p>
      {items.length ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {items.map((item) => (
            <span
              key={`${label}-${item}`}
              className="metric-mono border border-accent/25 bg-accent/8 px-2 py-1 text-[11px] text-accent"
            >
              {item}
            </span>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-xs text-muted">{emptyLabel}</p>
      )}
    </div>
  );
}

function GovernancePanel({
  topology,
  maskedHeads,
  state,
  selectedLayer,
  onStateChange,
  appendLog,
  syncDefects,
  isSyncingDefects,
}: {
  topology: ModelTopology | null;
  maskedHeads: StateResponse["masked_heads"];
  state: StateResponse | null;
  selectedLayer: LayerActivation | null;
  onStateChange: (state: StateResponse) => void;
  appendLog: (channel: string, message: string, detail?: string) => void;
  syncDefects?: () => Promise<void>;
  isSyncingDefects?: boolean;
}) {
  syncDefects = syncDefects ?? (async () => {});
  isSyncingDefects = isSyncingDefects ?? false;
  const [evidence, setEvidence] = useState<any | null>(null);
  const [evidenceTableFqn, setEvidenceTableFqn] = useState<string | null>(null);
  const [isFetchingEvidence, setIsFetchingEvidence] = useState(false);
  const selectedLayerMaskedHeads = selectedLayer
    ? maskedHeads
        .filter((mask) => mask.layer_index === selectedLayer.layer_index)
        .map((mask) => mask.head_name)
    : [];
  const topHead = selectedLayer?.top_heads.find((head) => !head.masked) ?? selectedLayer?.top_heads[0] ?? null;

  async function handleMaskSelectedHead() {
    if (!selectedLayer || !topHead) {
      return;
    }
    try {
      const nextState = await setLocalHeadMask(selectedLayer.layer_index, topHead.head_index);
      onStateChange(nextState);
      appendLog(
        "MASK",
        `Locally quarantined ${selectedLayer.layer_name}:${topHead.head_name}.`,
        "Run another faithful probe to see this head masked in the trace.",
      );
    } catch (maskError) {
      appendLog(
        "ERROR",
        "Local quarantine failed.",
        maskError instanceof Error ? maskError.message : String(maskError),
      );
    }
  }

  async function handleClearMasks() {
    try {
      const nextState = await clearLocalHeadMasks();
      onStateChange(nextState);
      appendLog("MASK", "Cleared all local quarantined heads.");
    } catch (maskError) {
      appendLog(
        "ERROR",
        "Failed to clear local masks.",
        maskError instanceof Error ? maskError.message : String(maskError),
      );
    }
  }

  async function handleFetchEvidence() {
    if (!selectedLayer) return;
    setIsFetchingEvidence(true);
    setEvidence(null);
    try {
      // Try to fetch catalog detail and resolve a matching table FQN for the selected layer
      let tableFqn: string | null = null;
      try {
        const catalog = await fetchCatalogDetail();
        const layerTables = Array.isArray(catalog.layer_tables) ? catalog.layer_tables : [];
        const entry = layerTables.find((lt: any) => lt.layer_index === selectedLayer.layer_index);
        if (entry && entry.table_fqn) tableFqn = entry.table_fqn;
      } catch (e) {
        // ignore and fall back to heuristic
      }

      if (!tableFqn) {
        // Best-effort fallback using common runtime defaults
        const serviceName = "Synapse_Neural_Service";
        const schemaName = "Transformer_Graph";
        const dbName = state?.topology?.model_name ?? "gpt2";
        tableFqn = `${serviceName}.${dbName}.${schemaName}.${selectedLayer.layer_name}`;
      }
      setEvidenceTableFqn(tableFqn);

      const parsed = await fetchParsedEvidence(tableFqn);
      setEvidence(parsed);
      appendLog("OM", `Fetched parsed evidence for ${tableFqn}`);
    } catch (err) {
      appendLog("ERROR", "Failed to fetch parsed evidence.", err instanceof Error ? err.message : String(err));
    } finally {
      setIsFetchingEvidence(false);
    }
  }

  // Auto-fetch evidence when the selected layer changes so the UI always
  // shows the latest parsed SYNAPSE_META for the chosen layer.
  useEffect(() => {
    if (!selectedLayer) {
      setEvidence(null);
      setEvidenceTableFqn(null);
      return;
    }
    void handleFetchEvidence();
  }, [selectedLayer?.layer_index]);

  return (
    <div className="panel-shell rounded-sm p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="panel-label">Governance / Quarantine</p>
          <h3 className="mt-2 text-lg font-medium text-zinc-50">Defective head status</h3>
        </div>
        <div className="metric-mono text-right text-xs text-muted">
          <p>{state?.openmetadata.connected ? "OpenMetadata online" : "OpenMetadata offline"}</p>
          <p>{topology?.model_name ?? "Unknown model"}</p>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        <div className="flex justify-end">
          <button
            type="button"
            onClick={async () => {
              appendLog("SYNC", "Syncing defects from OpenMetadata...");
              try {
                await syncDefects?.();
                appendLog("SYNC", "Sync defects completed.");
              } catch (e) {
                appendLog("ERROR", "Sync defects failed.", String(e));
              }
            }}
            disabled={isSyncingDefects}
            className="rounded-sm border border-accent/30 bg-accent/8 px-3 py-2 text-xs text-accent disabled:cursor-not-allowed disabled:border-line disabled:bg-panel2 disabled:text-muted"
          >
            {isSyncingDefects ? "Syncing..." : "Sync Defects"}
          </button>
        </div>
        {state?.openmetadata.last_ingest_error ? (
          <div className="border border-rose-500/30 bg-rose-500/10 p-3">
            <p className="panel-label text-rose-300">OpenMetadata Error</p>
            <p className="mt-2 text-xs leading-6 text-rose-200">
              {state.openmetadata.last_ingest_error}
            </p>
          </div>
        ) : null}

        <div className="border border-line bg-panel2/70 p-3">
          <p className="panel-label">Selected Layer</p>
          <p className="metric-mono mt-2 text-sm text-zinc-100">
            {selectedLayer?.layer_name ?? "No layer selected"}
          </p>
          <p className="mt-2 text-xs text-muted">
            {selectedLayer?.masked_head_names.length
              ? `Masked in this displayed trace: ${selectedLayer.masked_head_names.join(", ")}`
              : selectedLayerMaskedHeads.length
                ? `Queued for next trace in this layer: ${selectedLayerMaskedHeads.join(", ")}`
                : "No masked heads reported in the current layer."}
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleMaskSelectedHead}
              disabled={!selectedLayer || !topHead}
              className="rounded-sm border border-accent/30 bg-accent/8 px-3 py-2 text-xs text-accent transition hover:border-accent disabled:cursor-not-allowed disabled:border-line disabled:bg-panel disabled:text-muted"
            >
              Quarantine Top Head
            </button>
            <button
              type="button"
              onClick={handleClearMasks}
              disabled={maskedHeads.length === 0}
              className="rounded-sm border border-line bg-panel px-3 py-2 text-xs text-zinc-200 transition hover:border-accent disabled:cursor-not-allowed disabled:text-muted"
            >
              Clear Local Masks
            </button>
          </div>
          <p className="mt-3 text-xs leading-5 text-muted">
            This panel shows heads tagged as `DEFECTIVE` in OpenMetadata or local demo masks.
            Sync Defects pulls OpenMetadata tags into the runtime mask list; masking appears in
            the next faithful trace.
          </p>
        </div>

        <div className="thin-scrollbar max-h-60 space-y-2 overflow-y-auto border border-line bg-panel2/60 p-3 font-mono text-xs text-zinc-300">
          {/* OpenMetadata-tagged defective heads */}
          <div className="border border-line bg-panel/70 p-2">
            <p className="text-zinc-100">OpenMetadata defective heads: {state?.openmetadata.defective_heads?.length ?? 0}</p>
            <p className="mt-1 leading-5 text-muted">Heads tagged as `DEFECTIVE` in your OpenMetadata catalog.</p>
            {state?.openmetadata.defective_heads?.length ? (
              <div className="mt-2 flex gap-2 flex-wrap">
                {state?.openmetadata.defective_heads.map((h) => (
                  <span key={`${h.layer_index}-${h.head_index}`} className="metric-mono border border-accent/25 bg-accent/8 px-2 py-1 text-[11px] text-accent">
                    {h.layer_name}:{h.head_name}
                  </span>
                ))}
              </div>
            ) : null}
          </div>

          <div className="border border-line bg-panel/70 p-2">
            <p className="text-zinc-100">Runtime mask list: {maskedHeads.length}</p>
            <p className="mt-1 leading-5 text-muted">
              A nonzero count means those heads are queued for masking on the next traced run.
              The current trace only shows them after you run again.
            </p>
          </div>

          {maskedHeads.length === 0 ? (
            <p className="leading-5 text-muted">
              No heads are currently quarantined. Tag a head/layer as `DEFECTIVE` in
              OpenMetadata, sync defects, or use Quarantine Top Head for a local demo mask.
            </p>
          ) : null}
          <div className="mt-3 border border-line bg-panel/70 p-2">
            <div className="flex items-center justify-between">
              <p className="text-zinc-100">Lineage Evidence</p>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleFetchEvidence()}
                  disabled={!selectedLayer || isFetchingEvidence}
                  className="rounded-sm border border-accent/30 bg-accent/8 px-2 py-1 text-xs text-accent disabled:cursor-not-allowed disabled:border-line disabled:bg-panel2 disabled:text-muted"
                >
                  {isFetchingEvidence ? "Fetching..." : "Fetch Evidence"}
                </button>
              </div>
            </div>
            <p className="mt-1 text-xs text-muted">Retrieve parsed `SYNAPSE_META` evidence for the selected layer from OpenMetadata.</p>
            {evidence ? (
              <div className="mt-2 space-y-2">
                <p className="text-xs text-muted">Queried table: {evidence.table_fqn ?? evidenceTableFqn}</p>
                {evidence.edges?.length ? (
                  evidence.edges.map((edge: any, idx: number) => (
                    <div key={idx} className="p-3 bg-panel border border-line rounded-sm flex items-start gap-4">
                      <div className="flex-1">
                        <p className="text-sm text-accent">{edge.from ?? 'unknown'} → {edge.to ?? 'unknown'}</p>
                        {edge.synapse_meta ? (
                          <div className="mt-2 text-xs text-muted space-y-1">
                            <div><strong>session:</strong> {edge.synapse_meta.session_id}</div>
                            <div><strong>step:</strong> {String(edge.synapse_meta.step_index)}</div>
                            <div><strong>activation_path:</strong> {Array.isArray(edge.synapse_meta.activation_path) ? edge.synapse_meta.activation_path.join(' → ') : String(edge.synapse_meta.activation_path)}</div>
                            <div><strong>evidence_tokens:</strong> {Array.isArray(edge.synapse_meta.evidence_tokens) ? edge.synapse_meta.evidence_tokens.join(', ') : String(edge.synapse_meta.evidence_tokens)}</div>
                            {edge.synapse_meta.explanation ? (
                              <div className="mt-1"><strong>explanation:</strong> {truncateText(edge.synapse_meta.explanation, 220)}</div>
                            ) : null}
                          </div>
                        ) : (
                          <p className="mt-1 text-xs text-muted">No synapse_meta parsed from this edge.</p>
                        )}

                        {edge.columnsLineage ? (
                          <div className="mt-2 text-xs text-muted">
                            <strong>Columns Lineage:</strong>
                            {Array.isArray(edge.columnsLineage) && edge.columnsLineage.length ? (
                              <ul className="list-disc ml-4 mt-1">
                                {edge.columnsLineage.map((c: any, ci: number) => (
                                  <li key={ci} className="text-xs">
                                    <span className="font-mono">{c.toColumn}</span> ← {Array.isArray(c.fromColumns) ? c.fromColumns.join(', ') : String(c.fromColumns)}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <div className="mt-1 text-xs text-muted">None</div>
                            )}
                          </div>
                        ) : null}
                      </div>

                      <div className="flex flex-col gap-2">
                        <button
                          type="button"
                          onClick={async () => {
                            try {
                              const payload = JSON.stringify(edge.synapse_meta ?? edge, null, 2);
                              await navigator.clipboard.writeText(payload);
                              appendLog('OM', 'Copied evidence to clipboard');
                            } catch (e) {
                              appendLog('ERROR', 'Copy failed', String(e));
                            }
                          }}
                          className="rounded-sm border border-line px-2 py-1 text-xs"
                        >
                          Copy
                        </button>
                        <button
                          type="button"
                          onClick={async () => {
                            try {
                              const payload = JSON.stringify(edge, null, 2);
                              await navigator.clipboard.writeText(payload);
                              appendLog('OM', 'Copied raw edge to clipboard');
                            } catch (e) {
                              appendLog('ERROR', 'Copy failed', String(e));
                            }
                          }}
                          className="rounded-sm border border-line px-2 py-1 text-xs"
                        >
                          Copy Raw
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="mt-2 text-xs text-muted">No lineage edges returned for this layer.</p>
                )}
              </div>
            ) : null}
          </div>
          {maskedHeads.map((mask) => (
            <div key={`${mask.layer_index}-${mask.head_index}`} className="border border-line p-2">
              <p className="text-accent">
                {mask.layer_name}:{mask.head_name}
              </p>
              <p className="mt-1 text-muted">{mask.reason ?? "Tagged as DEFECTIVE in OpenMetadata."}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function upsertTrace(
  currentTrace: AttentionTrace | null,
  step: TokenStepCapture,
  prompt: string,
  topology: ModelTopology | null,
  executionMode: TraceExecutionMode,
  ollamaAvailable: boolean,
): AttentionTrace {
  if (!currentTrace) {
    return {
      source_prompt: prompt,
      generation_model: topology?.model_name ?? "local-model",
      generation_backend: resolvePredictedBackend(executionMode, ollamaAvailable),
      analysis_model:
        executionMode === "faithful" || !ollamaAvailable
          ? topology?.model_name ?? "local-model"
          : "shadow-model",
      analysis_mode: resolvePredictedAnalysisMode(executionMode, ollamaAvailable),
      trace_fidelity: resolvePredictedFidelity(executionMode, ollamaAvailable),
      prompt_token_count: 1,
      generated_text: step.generated_token,
      analysis_error: null,
      summary: null,
      steps: [step],
    };
  }

  return {
    ...currentTrace,
    generated_text: `${currentTrace.generated_text}${step.generated_token}`,
    steps: [...currentTrace.steps, step],
  };
}

function emptyState(topology: ModelTopology | null): StateResponse {
  return {
    topology,
    latest_session: null,
    masked_heads: [],
    ollama_available: false,
    openmetadata: {
      enabled: true,
      connected: false,
      catalog_ready: false,
      defective_heads: [],
      last_defect_sync_at: null,
      last_ingest_error: null,
    },
  };
}

function resolvePredictedBackend(
  executionMode: TraceExecutionMode,
  ollamaAvailable: boolean,
): AttentionTrace["generation_backend"] {
  if (executionMode === "faithful" || !ollamaAvailable) {
    return "huggingface";
  }
  return "ollama";
}

function resolvePredictedAnalysisMode(
  executionMode: TraceExecutionMode,
  ollamaAvailable: boolean,
): AttentionTrace["analysis_mode"] {
  if (executionMode === "faithful" || !ollamaAvailable) {
    return "inline";
  }
  return "shadow";
}

function resolvePredictedFidelity(
  executionMode: TraceExecutionMode,
  ollamaAvailable: boolean,
): TraceFidelity {
  if (executionMode === "faithful" || !ollamaAvailable) {
    return "exact";
  }
  return "proxy";
}

function predictedAnalysisMode(
  executionMode: TraceExecutionMode,
  ollamaAvailable: boolean,
): string {
  return resolvePredictedAnalysisMode(executionMode, ollamaAvailable);
}

function backendLabel(backend: AttentionTrace["generation_backend"]): string {
  return backend === "ollama" ? "Ollama live" : "HF inline";
}

function fidelityLabel(fidelity: TraceFidelity): string {
  return fidelity === "exact" ? "Exact evidence" : "Proxy evidence";
}

function executionModeLabel(mode: TraceExecutionMode): string {
  switch (mode) {
    case "faithful":
      return "Faithful glassbox";
    case "fast":
      return "Fast shadow trace";
    default:
      return "Auto orchestration";
  }
}

function modeRuntimeHint(mode: TraceExecutionMode, ollamaAvailable: boolean): string {
  if (mode === "faithful") {
    return "Always exact";
  }
  if (ollamaAvailable) {
    return mode === "fast" ? "Ollama + shadow" : "Ollama preferred";
  }
  return "HF fallback";
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1)}…`;
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function mergeHeadMasks(...maskGroups: Array<StateResponse["masked_heads"]>): StateResponse["masked_heads"] {
  const merged = new Map<string, StateResponse["masked_heads"][number]>();
  for (const group of maskGroups) {
    for (const mask of group) {
      merged.set(`${mask.layer_index}:${mask.head_index}`, mask);
    }
  }
  return Array.from(merged.values()).sort(
    (left, right) => left.layer_index - right.layer_index || left.head_index - right.head_index,
  );
}
