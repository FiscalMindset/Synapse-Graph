"use client";

import { useEffect, useState } from "react";
import { Calendar, Play, Zap, BarChart3, ChevronDown } from "lucide-react";
import {
  fetchSessionList,
  fetchSession,
  replaySession,
  type SessionSummary,
} from "@/lib/api";
import type { AttentionTrace } from "@/lib/types";

interface SessionDetail extends SessionSummary {
  response_text: string;
  trace: AttentionTrace;
  session_meta?: {
    _synapse_session_meta?: {
      generation_seed: number;
      generation_model: string;
      timestamp: string;
    };
  };
}

interface SessionTimelineProps {
  onSessionLoad?: (session: SessionDetail) => void;
  onReplayStart?: () => void;
  onReplayDone?: (response: any) => void;
}

export function SessionTimeline({ onSessionLoad, onReplayStart, onReplayDone }: SessionTimelineProps) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSession, setSelectedSession] = useState<SessionDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSessionId, setExpandedSessionId] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  async function loadSessions() {
    setIsLoading(true);
    setError(null);
    try {
      const list = await fetchSessionList();
      setSessions(list);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sessions");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleSessionClick(sessionId: string) {
    setIsLoading(true);
    setError(null);
    try {
      const detail = await fetchSession(sessionId);
      setSelectedSession(detail);
      setExpandedSessionId(sessionId);
      onSessionLoad?.(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load session");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleReplay(sessionId: string) {
    setIsReplaying(true);
    setError(null);
    onReplayStart?.();
    try {
      const response = await replaySession(sessionId, false);
      onReplayDone?.(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Replay failed");
    } finally {
      setIsReplaying(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="panel-label">Session Archive</p>
          <h2 className="mt-2 text-lg font-medium text-zinc-50">Browse past generations</h2>
        </div>
        <button
          onClick={loadSessions}
          disabled={isLoading}
          className={`border border-line bg-panel2 px-3 py-2 text-xs uppercase tracking-[0.22em] transition ${
            isLoading ? "text-muted cursor-not-allowed" : "text-zinc-200 hover:border-accent hover:text-accent"
          }`}
        >
          {isLoading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="rounded border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      <div className="space-y-3">
        {sessions.length === 0 ? (
          <div className="rounded border border-line bg-panel2 p-6 text-center text-muted">
            <p>No sessions recorded yet. Run a generation to create a session.</p>
          </div>
        ) : (
          sessions.map((session) => (
            <div
              key={session.session_id}
              className="rounded border border-line bg-panel2 transition hover:border-accent/50"
            >
              <button
                onClick={() => handleSessionClick(session.session_id)}
                className="w-full px-4 py-3 text-left hover:bg-panel1/50"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-accent">
                        {session.session_id.slice(0, 8)}
                      </span>
                      <span className="text-xs text-muted">
                        <Calendar className="inline h-3 w-3 mr-1" />
                        {formatDate(session.created_at)}
                      </span>
                    </div>
                    <p className="text-sm leading-5 text-zinc-300 line-clamp-2">
                      {session.prompt}
                    </p>
                    <p className="text-xs text-muted line-clamp-1">
                      {session.response_text_preview}
                    </p>
                  </div>
                  <ChevronDown
                    className={`h-4 w-4 text-muted transition ${
                      expandedSessionId === session.session_id ? "rotate-180" : ""
                    }`}
                  />
                </div>
              </button>

              {expandedSessionId === session.session_id && selectedSession && (
                <div className="border-t border-line bg-panel1/30 px-4 py-4 space-y-4">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded bg-panel2 p-3">
                      <div className="flex items-center gap-2 text-xs text-muted">
                        <Zap className="h-4 w-4" />
                        Trace Steps
                      </div>
                      <p className="mt-1 text-2xl font-semibold text-zinc-50">
                        {selectedSession.trace?.steps?.length ?? 0}
                      </p>
                    </div>
                    <div className="rounded bg-panel2 p-3">
                      <div className="flex items-center gap-2 text-xs text-muted">
                        <BarChart3 className="h-4 w-4" />
                        Model
                      </div>
                      <p className="mt-1 text-sm font-semibold text-zinc-50">
                        {selectedSession.trace?.generation_model ?? "Unknown"}
                      </p>
                    </div>
                  </div>

                  {selectedSession.session_meta?._synapse_session_meta && (
                    <div className="rounded bg-panel2 p-3 space-y-2 text-xs">
                      <div>
                        <span className="text-muted">Seed:</span>{" "}
                        <span className="font-mono text-zinc-300">
                          {selectedSession.session_meta._synapse_session_meta.generation_seed}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted">Timestamp:</span>{" "}
                        <span className="text-zinc-300">
                          {new Date(
                            selectedSession.session_meta._synapse_session_meta.timestamp,
                          ).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2">
                    <button
                      onClick={() => handleReplay(session.session_id)}
                      disabled={isReplaying}
                      className={`flex-1 rounded border px-3 py-2 text-sm transition flex items-center justify-center gap-2 ${
                        isReplaying
                          ? "border-muted text-muted cursor-not-allowed bg-muted/5"
                          : "border-accent text-accent hover:bg-accent/10"
                      }`}
                    >
                      <Play className="h-4 w-4" />
                      {isReplaying ? "Replaying..." : "Replay Session"}
                    </button>
                    <button
                      onClick={() => {
                        setExpandedSessionId(null);
                        setSelectedSession(null);
                      }}
                      className="flex-1 rounded border border-line px-3 py-2 text-sm text-zinc-300 hover:border-accent/50 transition"
                    >
                      Close
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function formatDate(dateString: string): string {
  try {
    const date = new Date(dateString);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "Unknown date";
  }
}
