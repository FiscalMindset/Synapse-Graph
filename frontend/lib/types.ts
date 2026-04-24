export type GenerationBackend = "ollama" | "huggingface";
export type AnalysisMode = "inline" | "shadow";

export interface HeadMask {
  layer_index: number;
  layer_name: string;
  head_index: number;
  head_name: string;
  reason?: string | null;
}

export interface HeadActivation {
  head_index: number;
  head_name: string;
  masked: boolean;
  max_attention_score: number;
  mean_attention_score: number;
  l2_norm: number;
  top_source_positions: number[];
  raw_last_token_attention: number[];
}

export interface LayerActivation {
  layer_index: number;
  layer_name: string;
  sequence_length: number;
  head_count: number;
  masked_head_names: string[];
  top_heads: HeadActivation[];
  full_last_token_attention_matrix?: number[][] | null;
}

export interface TokenStepCapture {
  step_index: number;
  generated_token: string;
  generated_token_id: number;
  prompt_plus_generation_length: number;
  masked_heads: string[];
  layers: LayerActivation[];
  high_activation_path: string[];
}

export interface AttentionTrace {
  source_prompt: string;
  generation_model: string;
  analysis_model: string;
  generation_backend: GenerationBackend;
  analysis_mode: AnalysisMode;
  prompt_token_count: number;
  generated_text: string;
  analysis_error?: string | null;
  steps: TokenStepCapture[];
}

export interface LayerTopology {
  layer_index: number;
  layer_name: string;
  head_count: number;
  head_dim: number;
}

export interface ModelTopology {
  model_name: string;
  device: string;
  total_layers: number;
  total_heads: number;
  layers: LayerTopology[];
}

export interface OpenMetadataStatus {
  enabled: boolean;
  connected: boolean;
  catalog_ready: boolean;
  defective_heads: HeadMask[];
  last_defect_sync_at?: string | null;
  last_ingest_error?: string | null;
}

export interface SessionSnapshot {
  session_id: string;
  created_at: string;
  prompt: string;
  response_text: string;
  trace: AttentionTrace;
  masked_heads: HeadMask[];
}

export interface StateResponse {
  topology?: ModelTopology | null;
  latest_session?: SessionSnapshot | null;
  masked_heads: HeadMask[];
  ollama_available: boolean;
  openmetadata: OpenMetadataStatus;
}

export interface GeneratePayload {
  prompt: string;
  system_prompt?: string | null;
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  stop: string[];
  stream: boolean;
}

export interface StreamSessionEvent {
  sessionId: string;
  topology: ModelTopology;
  maskedHeads: HeadMask[];
  openmetadata: OpenMetadataStatus;
}

export interface StreamTokenEvent {
  sessionId: string;
  token: string;
}

export interface StreamTraceStepEvent {
  sessionId: string;
  step: TokenStepCapture;
}

export interface StreamDoneEvent {
  sessionId: string;
  responseText: string;
  trace: AttentionTrace;
  maskedHeads: HeadMask[];
}

export interface StreamErrorEvent {
  sessionId: string;
  message: string;
}

export interface LogEntry {
  id: string;
  channel: string;
  message: string;
  detail?: string;
  createdAt: string;
}
