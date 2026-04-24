"use client";

import { startTransition, useDeferredValue, useEffect, useState } from "react";
import { Activity, Orbit, Radio, ShieldAlert } from "lucide-react";

import { ActivationChart } from "@/components/activation-chart";
import { ConsoleLog } from "@/components/console-log";
import { SynapseGraph } from "@/components/synapse-graph";
import { fetchState, streamGeneration, syncOpenMetadataDefects, preloadHF } from "@/lib/api";
import type {
  AttentionTrace,
  LayerActivation,
  LogEntry,
  ModelTopology,
  StateResponse,
  StreamDoneEvent,
  TokenStepCapture,
} from "@/lib/types";

const DEFAULT_PROMPT =
  "Trace the attention route you would use to explain why masking a single head can change a model's response.";

export function SynapseDashboard() {
  const [state, setState] = useState<StateResponse | null>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [systemPrompt, setSystemPrompt] = useState(
    "You are the instrumented local model inside Synapse-Graph. Explain your reasoning concisely.",
  );
  const [responseText, setResponseText] = useState("");
  const [trace, setTrace] = useState<AttentionTrace | null>(null);
  const [selectedLayerIndex, setSelectedLayerIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const deferredTrace = useDeferredValue(trace);
  const topology = state?.topology ?? null;
  const maskedHeads = state?.masked_heads ?? [];
  const latestStep = deferredTrace?.steps.at(-1) ?? null;
  const selectedLayer = latestStep?.layers.find((layer) => layer.layer_index === selectedLayerIndex) ?? null;

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
    try {
      const nextState = await syncOpenMetadataDefects();
      setState(nextState);
      appendLog(
        "SYNC",
        `Synchronized ${nextState.masked_heads.length} defective heads from OpenMetadata.`,
      );
    } catch (syncError) {
      const message = syncError instanceof Error ? syncError.message : "Defect sync failed.";
      setError(message);
      appendLog("ERROR", "OpenMetadata defect sync failed.", message);
    }
  }

  async function handleProbe() {
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
          max_new_tokens: 160,
          temperature: 0.2,
          top_p: 0.95,
          stop: [],
          stream: true,
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
              `Session ${event.sessionId.slice(0, 8)} opened with ${event.topology.total_layers} traced layers.`,
            );
          },
          onToken: (event) => {
            setResponseText((current) => current + event.token);
          },
          onTraceStep: (event) => {
            startTransition(() => {
              setTrace((current) => upsertTrace(current, event.step, prompt, topology));
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
                Local Ollama generation stays in the driver seat while an instrumented PyTorch
                shadow model emits per-head attention telemetry into OpenMetadata.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                icon={Orbit}
                label="Generation Backend"
                value={state?.ollama_available ? "Ollama live" : "HF fallback"}
                detail={topology?.device ?? "Awaiting topology"}
              />
              <MetricCard
                icon={Radio}
                label="Lineage Depth"
                value={`${latestStep?.high_activation_path.length ?? 0} active hops`}
                detail={trace?.analysis_mode ?? "No trace"}
              />
              <MetricCard
                icon={ShieldAlert}
                label="Masked Heads"
                value={`${maskedHeads.length}`}
                detail={state?.openmetadata.connected ? "OM synchronized" : "Local only"}
              />
              <MetricCard
                icon={Activity}
                label="Response Tokens"
                value={`${responseText.length}`}
                detail={isRunning ? "Streaming" : "Idle"}
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
                  className="border border-line bg-panel2 px-3 py-2 text-xs uppercase tracking-[0.22em] text-zinc-200 transition hover:border-accent hover:text-accent"
                >
                  Sync Defects
                </button>
              </div>

              <div className="mt-5 space-y-4">
                <label className="block">
                  <span className="panel-label">System Prompt</span>
                  <textarea
                    value={systemPrompt}
                    onChange={(event) => setSystemPrompt(event.target.value)}
                    className="mt-2 h-28 w-full resize-none rounded-sm border border-line bg-panel2 px-3 py-3 text-sm text-zinc-100 outline-none transition focus:border-accent"
                  />
                </label>

                <label className="block">
                  <span className="panel-label">User Prompt</span>
                  <textarea
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    className="mt-2 h-40 w-full resize-none rounded-sm border border-line bg-panel2 px-3 py-3 text-sm text-zinc-100 outline-none transition focus:border-accent"
                  />
                </label>

                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handleProbe}
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
              maskedHeads={maskedHeads}
              onSelectLayer={setSelectedLayerIndex}
            />
          </section>

          <section className="space-y-5">
            <ActivationChart layer={selectedLayer} />

            <ResponsePanel responseText={responseText} latestStep={latestStep} error={error} />

            <GovernancePanel
              topology={topology}
              maskedHeads={maskedHeads}
              state={state}
              selectedLayer={selectedLayer}
            />
          </section>
        </div>
      </div>
    </main>
  );

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
    <div className="panel-shell rounded-sm p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="panel-label">Response Stream</p>
          <h3 className="mt-2 text-lg font-medium text-zinc-50">Generation output</h3>
        </div>
        <div className="metric-mono text-right text-xs text-muted">
          <p>{latestStep ? `step ${latestStep.step_index}` : "No trace"}</p>
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

function GovernancePanel({
  topology,
  maskedHeads,
  state,
  selectedLayer,
}: {
  topology: ModelTopology | null;
  maskedHeads: StateResponse["masked_heads"];
  state: StateResponse | null;
  selectedLayer: LayerActivation | null;
}) {
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
              ? `Masked in layer: ${selectedLayer.masked_head_names.join(", ")}`
              : "No masked heads reported in the current layer."}
          </p>
        </div>

        <div className="thin-scrollbar max-h-60 space-y-2 overflow-y-auto border border-line bg-panel2/60 p-3 font-mono text-xs text-zinc-300">
          {maskedHeads.length === 0 ? (
            <p className="text-muted">No heads are currently quarantined.</p>
          ) : null}
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
): AttentionTrace {
  if (!currentTrace) {
    return {
      source_prompt: prompt,
      generation_model: topology?.model_name ?? "local-model",
      analysis_model: topology?.model_name ?? "local-model",
      generation_backend: "huggingface",
      analysis_mode: "shadow",
      prompt_token_count: 0,
      generated_text: "",
      analysis_error: null,
      steps: [step],
    };
  }

  return {
    ...currentTrace,
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
