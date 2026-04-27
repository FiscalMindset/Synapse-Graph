export type GenerationBackend = "ollama" | "huggingface";
export type AnalysisMode = "inline" | "shadow";
export type TraceExecutionMode = "auto" | "fast" | "faithful";
export type TraceFidelity = "exact" | "proxy";

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
  top_source_tokens: string[];
  raw_last_token_attention: number[];
}

export interface LayerActivation {
  layer_index: number;
  layer_name: string;
  sequence_length: number;
  head_count: number;
  masked_head_names: string[];
  dominant_source_tokens: string[];
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
  evidence_tokens: string[];
  evidence_positions?: number[];
  evidence_token_ids?: number[];
  evidence_token_attention?: Record<number, number>;
  explanation?: string | null;
}

export interface TraceSummary {
  explanation: string;
  dominant_layers: string[];
  dominant_heads: string[];
  influential_tokens: string[];
  masked_heads_applied: string[];
}

export interface EvidenceQuality {
  score: number;
  label: string;
  exactness: string;
  causal_validation: string;
  black_box_gaps: string[];
  recommended_next_actions: string[];
}

export interface AttentionTrace {
  source_prompt: string;
  generation_model: string;
  analysis_model: string;
  generation_backend: GenerationBackend;
  analysis_mode: AnalysisMode;
  trace_fidelity: TraceFidelity;
  match_score?: number | null;
  fidelity_reason?: string | null;
  evidence_quality?: EvidenceQuality | null;
  prompt_token_count: number;
  generated_text: string;
  analysis_error?: string | null;
  summary?: TraceSummary | null;
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

export interface InferenceResponse {
  backend: GenerationBackend;
  generation_model: string;
  analysis_model: string;
  text: string;
  trace: AttentionTrace;
}

export interface CircuitHead {
  layer_index: number;
  layer_name: string;
  head_index: number;
  head_name: string;
  activation_score: number;
}

export interface CircuitAblationResult {
  masked_heads: CircuitHead[];
  output_text: string;
  target_present: boolean;
  target_count: number;
  text_similarity: number;
  causal_effect_score: number;
  trace?: AttentionTrace | null;
}

export interface CircuitDiscoveryRequest {
  prompt: string;
  target_hallucination_token: string;
  system_prompt?: string | null;
  trace_model_name?: string | null;
  max_new_tokens?: number;
  top_k_heads?: number;
  max_pair_sweeps?: number;
}

export interface CircuitDiscoveryResponse {
  target_hallucination_token: string;
  baseline: InferenceResponse;
  baseline_target_present: boolean;
  baseline_target_count: number;
  candidate_heads: CircuitHead[];
  sweep_results: CircuitAblationResult[];
  discovered_circuit: CircuitHead[];
  combined_causal_effect: number;
  verdict: string;
}

export interface CausalAutopsyRequest {
  prompt: string;
  system_prompt?: string | null;
  trace_model_name?: string | null;
  max_new_tokens?: number;
  layer_index?: number | null;
  head_index?: number | null;
}

export interface CausalAutopsyResponse {
  target: CircuitHead;
  baseline: InferenceResponse;
  ablated: InferenceResponse;
  text_similarity: number;
  causal_effect_score: number;
  verdict: string;
  interpretation: string;
}

export interface QuarantineRequest {
  heads: CircuitHead[];
  reason?: string | null;
}

export interface OpenMetadataWebhookResponse {
  applied: boolean;
  parsed_heads: HeadMask[];
  masked_heads: HeadMask[];
  reason: string;
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
  execution_mode: TraceExecutionMode;
  trace_model_name?: string | null;
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
