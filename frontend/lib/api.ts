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
