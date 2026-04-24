"use client";

import type { LogEntry } from "@/lib/types";

interface ConsoleLogProps {
  logs: LogEntry[];
}

export function ConsoleLog({ logs }: ConsoleLogProps) {
  return (
    <div className="panel-shell thin-scrollbar flex h-[320px] flex-col overflow-hidden rounded-sm">
      <div className="border-b border-line px-4 py-3">
        <p className="panel-label">OpenMetadata / Proxy Log</p>
      </div>
      <div className="thin-scrollbar flex-1 space-y-3 overflow-y-auto px-4 py-4 font-mono text-xs text-zinc-300">
        {logs.length === 0 ? (
          <div className="rounded-sm border border-dashed border-line bg-panel2/60 p-3 text-muted">
            Awaiting telemetry stream.
          </div>
        ) : null}
        {logs.map((log) => (
          <div key={log.id} className="border border-line bg-panel2/70 p-3">
            <div className="flex items-center justify-between gap-3">
              <span className="text-[10px] uppercase tracking-[0.24em] text-accent">
                {log.channel}
              </span>
              <span className="text-[10px] text-muted">{log.createdAt}</span>
            </div>
            <p className="mt-2 text-zinc-100">{log.message}</p>
            {log.detail ? <pre className="mt-2 whitespace-pre-wrap text-muted">{log.detail}</pre> : null}
          </div>
        ))}
      </div>
    </div>
  );
}
