import type {
  GeneratePayload,
  StateResponse,
  StreamDoneEvent,
  StreamErrorEvent,
  StreamSessionEvent,
  StreamTokenEvent,
  StreamTraceStepEvent,
} from "@/lib/types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_NEURAL_PROXY_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export async function fetchState(): Promise<StateResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/state`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`State request failed: ${response.status}`);
  }

  return (await response.json()) as StateResponse;
}

export async function syncOpenMetadataDefects(): Promise<StateResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/openmetadata/sync-defects`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Defect sync failed: ${response.status}`);
  }

  return (await response.json()) as StateResponse;
}

export async function setLocalHeadMask(layerIndex: number, headIndex: number): Promise<StateResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/governance/local-mask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      layer_index: layerIndex,
      head_index: headIndex,
    }),
  });

  if (!response.ok) {
    throw new Error(`Local mask request failed: ${response.status}`);
  }

  return (await response.json()) as StateResponse;
}

export async function clearLocalHeadMasks(): Promise<StateResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/governance/clear-local-masks`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Clear local masks failed: ${response.status}`);
  }

  return (await response.json()) as StateResponse;
}

export async function preloadHF(): Promise<StateResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/hf/preload`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`HF preload request failed: ${response.status}`);
  }

  return (await response.json()) as StateResponse;
}

interface StreamHandlers {
  onSession?: (event: StreamSessionEvent) => void;
  onToken?: (event: StreamTokenEvent) => void;
  onTraceStep?: (event: StreamTraceStepEvent) => void;
  onDone?: (event: StreamDoneEvent) => void;
  onError?: (event: StreamErrorEvent) => void;
}

export async function streamGeneration(
  payload: GeneratePayload,
  handlers: StreamHandlers,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/generate/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Streaming request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const parsed = parseEventChunk(chunk);
      if (!parsed) {
        continue;
      }

      switch (parsed.event) {
        case "session":
          handlers.onSession?.(parsed.data as StreamSessionEvent);
          break;
        case "token":
          handlers.onToken?.(parsed.data as StreamTokenEvent);
          break;
        case "trace_step":
          handlers.onTraceStep?.(parsed.data as StreamTraceStepEvent);
          break;
        case "done":
          handlers.onDone?.(parsed.data as StreamDoneEvent);
          break;
        case "error":
          handlers.onError?.(parsed.data as StreamErrorEvent);
          break;
        default:
          break;
      }
    }
  }
}

export async function postCausalAutopsy(
  payload: Partial<import("./types").CausalAutopsyRequest>,
): Promise<import("./types").CausalAutopsyResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/autopsy/causal`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Causal autopsy failed: ${response.status}`);
  }

  return (await response.json()) as import("./types").CausalAutopsyResponse;
}

export async function postDiscoverCircuit(
  payload: import("./types").CircuitDiscoveryRequest,
): Promise<import("./types").CircuitDiscoveryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/autopsy/discover_circuit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Discover circuit failed: ${response.status}`);
  }

  return (await response.json()) as import("./types").CircuitDiscoveryResponse;
}

export async function postQuarantineCircuit(
  payload: import("./types").QuarantineRequest,
): Promise<import("./types").OpenMetadataWebhookResponse> {
  const controller = new AbortController();
  const timeoutMs = 15000; // 15s timeout for quick UI feedback
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/openmetadata/quarantine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Quarantine request failed: ${response.status}`);
    }

    return (await response.json()) as import("./types").OpenMetadataWebhookResponse;
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error("Quarantine request timed out");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export async function streamDiscoverCircuit(
  payload: import("./types").CircuitDiscoveryRequest,
  handlers: {
    onProgress?: (data: any) => void;
    onSweepStart?: (data: any) => void;
    onSweepDone?: (data: any) => void;
    onDone?: (data: any) => void;
    onError?: (data: any) => void;
  },
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/autopsy/discover_circuit/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Discover stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const parsed = parseEventChunk(chunk);
      if (!parsed) continue;
      try {
        switch (parsed.event) {
          case "progress":
            handlers.onProgress?.(parsed.data);
            break;
          case "sweep_start":
            handlers.onSweepStart?.(parsed.data);
            break;
          case "sweep_done":
            handlers.onSweepDone?.(parsed.data);
            break;
          case "done":
            handlers.onDone?.(parsed.data);
            break;
          case "error":
            handlers.onError?.(parsed.data);
            break;
          default:
            break;
        }
      } catch (e) {
        handlers.onError?.({ message: String(e) });
      }
    }
  }
}

export async function streamQuarantineCircuit(
  payload: import("./types").QuarantineRequest,
  handlers: {
    onProgress?: (data: any) => void;
    onDone?: (data: any) => void;
    onError?: (data: any) => void;
  },
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/openmetadata/quarantine/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Quarantine stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const parsed = parseEventChunk(chunk);
      if (!parsed) continue;
      try {
        switch (parsed.event) {
          case "progress":
            handlers.onProgress?.(parsed.data);
            break;
          case "done":
            handlers.onDone?.(parsed.data);
            break;
          case "error":
            handlers.onError?.(parsed.data);
            break;
          default:
            break;
        }
      } catch (e) {
        handlers.onError?.({ message: String(e) });
      }
    }
  }
}

function parseEventChunk(chunk: string): { event: string; data: unknown } | null {
  const lines = chunk.split("\n");
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
      continue;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  return {
    event: eventName,
    data: JSON.parse(dataLines.join("\n")),
  };
}
