"use client";

import { useEffect, useState } from "react";
import { ChevronDown, Zap, Target } from "lucide-react";
import { fetchParsedEvidence } from "@/lib/api";

interface EvidenceEdge {
  from: string;
  to: string;
  description: string | null;
  sqlQuery: string;
  columnsLineage: any[];
  synapse_meta?: {
    session_id: string;
    step_index: number;
    target: string;
    prompt_preview: string;
    activation_path: string[];
    evidence_tokens: string[];
    evidence_positions: number[];
    evidence_token_attention: Record<number, number>;
    explanation: string;
  };
}

interface EvidenceDrilldownProps {
  tableFqn: string;
  onEvidenceLoad?: (edges: EvidenceEdge[]) => void;
}

export function EvidenceDrilldown({ tableFqn, onEvidenceLoad }: EvidenceDrilldownProps) {
  const [edges, setEdges] = useState<EvidenceEdge[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedEdgeIdx, setExpandedEdgeIdx] = useState<number | null>(null);

  useEffect(() => {
    loadEvidence();
  }, [tableFqn]);

  async function loadEvidence() {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchParsedEvidence(tableFqn);
      const parsedEdges = data.edges ?? [];
      setEdges(parsedEdges);
      onEvidenceLoad?.(parsedEdges);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load evidence");
    } finally {
      setIsLoading(false);
    }
  }

  if (isLoading) {
    return (
      <div className="rounded border border-line bg-panel2 p-6 text-center text-muted">
        Loading evidence...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
        {error}
      </div>
    );
  }

  if (edges.length === 0) {
    return (
      <div className="rounded border border-line bg-panel2 p-6 text-center text-muted">
        <p>No lineage evidence available for this table.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {edges.map((edge, idx) => (
        <div
          key={idx}
          className="rounded border border-line bg-panel2 transition hover:border-accent/50"
        >
          <button
            onClick={() => setExpandedEdgeIdx(expandedEdgeIdx === idx ? null : idx)}
            className="w-full px-4 py-3 text-left hover:bg-panel1/50"
          >
            <div className="flex items-center justify-between gap-4">
              <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-zinc-300">{edge.from?.split(".").pop()}</span>
                  <span className="text-muted">→</span>
                  <span className="text-accent font-semibold">{edge.to?.split(".").pop()}</span>
                </div>
                {edge.synapse_meta && (
                  <div className="text-xs text-muted space-y-1">
                    <p>Step {edge.synapse_meta.step_index}: {edge.synapse_meta.explanation?.slice(0, 120)}...</p>
                    <div className="flex gap-2">
                      {edge.synapse_meta.evidence_tokens?.slice(0, 3).map((token, i) => (
                        <span key={i} className="px-2 py-0.5 rounded bg-accent/10 text-accent text-xs">
                          "{token}"
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <ChevronDown
                className={`h-4 w-4 text-muted transition ${
                  expandedEdgeIdx === idx ? "rotate-180" : ""
                }`}
              />
            </div>
          </button>

          {expandedEdgeIdx === idx && edge.synapse_meta && (
            <div className="border-t border-line bg-panel1/30 px-4 py-4 space-y-4">
              {/* Explanation */}
              <div>
                <h4 className="text-xs font-semibold text-zinc-300 mb-2">Explanation</h4>
                <p className="text-sm leading-5 text-muted">{edge.synapse_meta.explanation}</p>
              </div>

              {/* Activation Path */}
              {edge.synapse_meta.activation_path?.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300 mb-2">
                    <Zap className="h-3 w-3 text-accent" />
                    Activation Path
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {edge.synapse_meta.activation_path.map((layer, i) => (
                      <span
                        key={i}
                        className="px-2 py-1 rounded text-xs bg-accent/15 text-accent font-mono"
                      >
                        {layer}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Evidence Tokens with Attention Weights */}
              {edge.synapse_meta.evidence_tokens?.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300 mb-2">
                    <Target className="h-3 w-3 text-amber-400" />
                    Evidence Tokens
                  </div>
                  <div className="space-y-2">
                    {edge.synapse_meta.evidence_tokens.map((token, i) => {
                      const pos = edge.synapse_meta?.evidence_positions?.[i];
                      const attention = pos !== undefined ? edge.synapse_meta?.evidence_token_attention?.[pos] : 0;
                      return (
                        <div key={i} className="flex items-center gap-3">
                          <span className="font-mono text-xs px-2 py-1 rounded bg-panel2 text-zinc-300 flex-1">
                            "{token}"
                          </span>
                          {attention !== undefined && (
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-2 rounded bg-panel2 overflow-hidden">
                                <div
                                  className="h-full bg-accent transition-all"
                                  style={{
                                    width: `${Math.min(100, Math.max(0, attention * 100))}%`,
                                  }}
                                />
                              </div>
                              <span className="text-xs text-muted w-10 text-right">
                                {(attention * 100).toFixed(0)}%
                              </span>
                            </div>
                          )}
                          {pos !== undefined && (
                            <span className="text-xs text-muted">pos:{pos}</span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Evidence Positions */}
              {edge.synapse_meta.evidence_positions?.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold text-zinc-300 mb-2">Token Positions</h4>
                  <p className="text-xs text-muted font-mono">
                    {edge.synapse_meta.evidence_positions.join(", ")}
                  </p>
                </div>
              )}

              {/* Metadata */}
              <div className="rounded bg-panel2 p-3 space-y-1 text-xs">
                <div>
                  <span className="text-muted">Session:</span>{" "}
                  <span className="font-mono text-zinc-300">{edge.synapse_meta.session_id?.slice(0, 8)}</span>
                </div>
                <div>
                  <span className="text-muted">Step:</span>{" "}
                  <span className="font-mono text-zinc-300">{edge.synapse_meta.step_index}</span>
                </div>
                <div>
                  <span className="text-muted">Target:</span>{" "}
                  <span className="font-mono text-zinc-300">{edge.synapse_meta.target}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
