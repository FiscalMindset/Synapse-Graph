"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { LayerActivation } from "@/lib/types";

interface ActivationChartProps {
  layer: LayerActivation | null;
}

export function ActivationChart({ layer }: ActivationChartProps) {
  const data =
    layer?.top_heads.map((head) => ({
      name: head.head_name,
      score: Number(head.max_attention_score.toFixed(3)),
      masked: head.masked,
    })) ?? [];

  return (
    <div className="panel-shell rounded-sm p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="panel-label">Activation Lens</p>
          <h3 className="mt-2 text-lg font-medium text-zinc-50">
            {layer?.layer_name ?? "Select a layer"}
          </h3>
        </div>
        <div className="metric-mono text-right text-xs text-muted">
          <p>{layer ? `${layer.head_count} heads` : "No layer selected"}</p>
          <p>{layer ? `${layer.sequence_length} context` : "Awaiting trace"}</p>
        </div>
      </div>

      <div className="mt-5 h-64">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center border border-dashed border-line bg-panel2/70 text-sm text-muted">
            No head activations available for the current layer.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fill: "#7f848f", fontSize: 11, fontFamily: "var(--font-mono)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "#7f848f", fontSize: 11, fontFamily: "var(--font-mono)" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(57,255,20,0.05)" }}
                contentStyle={{
                  borderRadius: 2,
                  border: "1px solid #26282d",
                  background: "#0f1113",
                  color: "#fafafa",
                  fontFamily: "var(--font-mono)",
                }}
              />
              <Bar dataKey="score" radius={0}>
                {data.map((entry) => (
                  <Cell
                    key={entry.name}
                    fill={entry.masked ? "#52525b" : "#39ff14"}
                    fillOpacity={entry.masked ? 0.4 : 0.9}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
