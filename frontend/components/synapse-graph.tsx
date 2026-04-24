"use client";

import { useEffect, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";

import { buildSynapseGraph, type SynapseNodeData } from "@/lib/graph";
import type { AttentionTrace, HeadMask, ModelTopology } from "@/lib/types";

interface SynapseGraphProps {
  topology: ModelTopology | null | undefined;
  trace: AttentionTrace | null | undefined;
  maskedHeads: HeadMask[];
  onSelectLayer: (layerIndex: number) => void;
}

const nodeTypes = {
  synapse: SynapseNode,
};

export function SynapseGraph({
  topology,
  trace,
  maskedHeads,
  onSelectLayer,
}: SynapseGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 1200, height: 720 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0];
      setSize({ width: Math.max(320, r.contentRect.width), height: Math.max(200, r.contentRect.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  if (!topology) {
    return (
      <div ref={containerRef} className="panel-shell flex min-h-[320px] items-center justify-center rounded-sm text-muted">
        Loading transformer topology...
      </div>
    );
  }

  // Build the semantic nodes/edges then compute responsive positions based on container size.
  const { nodes: rawNodes, edges } = buildSynapseGraph(topology, trace, maskedHeads);

  const paddingX = 28;
  const paddingY = 20;
  const usableWidth = Math.max(320, size.width - paddingX * 2);
  // Allow the visualizer to expand to more columns on very wide screens so the
  // layout doesn't stay narrow in the middle of large viewports.
  const columns = Math.min(Math.max(1, Math.floor(usableWidth / 280)), 6);
  const rows = Math.ceil(Math.max(1, topology.layers.length) / columns);
  // Add a horizontal gap between columns so edge labels have breathing room.
  const columnGap = 36;
  const columnWidth = Math.max(180, (usableWidth - (columns - 1) * columnGap) / columns);
  const nodeWidth = Math.min(360, columnWidth - 48);
  // Increase minimum row height to provide clearer vertical spacing on large
  // graphs so the response node doesn't butt up against the last layer.
  const rowHeight = Math.max(220, Math.floor((size.height - 160) / Math.max(1, rows + 1)));

  const positionedNodes = rawNodes.map((n) => {
    if (n.id === "prompt") {
      const x = paddingX + (usableWidth - nodeWidth) / 2;
      const y = paddingY;
      return { ...n, position: { x, y } } as typeof n;
    }

    if (n.id === "response") {
      const x = paddingX + (usableWidth - nodeWidth) / 2;
      // add a bit more offset below the last row to avoid touching
      const responseYOffset = 36;
      const y = paddingY + (rows + 1) * rowHeight + responseYOffset;
      return { ...n, position: { x, y } } as typeof n;
    }

    if (n.id.startsWith("layer-")) {
      const layerIndex = Number(n.id.split("-")[1]);
      const posIndex = topology.layers.findIndex((l) => l.layer_index === layerIndex);
      const col = posIndex % columns;
      const row = Math.floor(posIndex / columns);
      const x = paddingX + col * columnWidth + (columnWidth - nodeWidth) / 2;
      const y = paddingY + (row + 1) * rowHeight;
      return { ...n, position: { x, y } } as typeof n;
    }

    return n;
  });

  return (
    <div ref={containerRef} className="panel-shell min-h-[360px] h-[60vh] md:h-[72vh] lg:h-[80vh] overflow-auto rounded-sm">
      <div className="border-b border-line px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="panel-label">Synapse Visualizer</p>
            <h2 className="mt-2 text-lg font-medium text-zinc-50">{topology.model_name}</h2>
          </div>
          <div className="metric-mono text-right text-xs text-muted">
            <p>{topology.total_layers} layers</p>
            <p>{topology.total_heads} total heads</p>
          </div>
        </div>
      </div>

      <ReactFlow<Node<SynapseNodeData>, Edge>
        nodes={positionedNodes}
        edges={edges}
        nodeTypes={nodeTypes}
        colorMode="dark"
        className="synapse-flow"
        fitView
        minZoom={0.12}
        maxZoom={1.6}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => {
          const data = node.data as unknown as SynapseNodeData | undefined;
          if (data?.kind === "layer" && typeof data.layerIndex === "number") {
            onSelectLayer(data.layerIndex);
          }
        }}
      >
        <Background gap={28} color="rgba(57,255,20,0.06)" variant={BackgroundVariant.Dots} size={1.2} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

function SynapseNode({ data }: NodeProps<Node<SynapseNodeData>>) {
  const isLayer = data.kind === "layer";

  return (
    <div
      style={{ backgroundColor: "var(--panel)" }}
      className={`synapse-node-panel min-w-[248px] border px-5 py-4 shadow-panel ${
        data.active ? "border-accent accent-glow" : "border-line"
      }`}
    >
      <Handle type="target" position={Position.Top} className="opacity-0" />
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="panel-label">{data.kind}</p>
          <h3 className="mt-2 text-base font-medium text-zinc-50">{data.label}</h3>
          <p className="mt-2 text-xs text-muted">{data.subtitle}</p>
        </div>
        {isLayer ? (
          <div className="metric-mono text-right text-[11px] text-muted">
            <p>{data.activeScore ? `max ${data.activeScore}` : "idle"}</p>
            <p>{data.maskedCount > 0 ? `${data.maskedCount} masked` : "clean"}</p>
          </div>
        ) : null}
      </div>

      {data.activeHeads.length > 0 ? (
        <div className="mt-4 flex flex-wrap gap-2">
        {data.activeHeads.map((activeHead) => (
          <span
            key={activeHead}
            style={{
              backgroundColor: "var(--panel-2)",
              maxWidth: 120,
              display: "inline-block",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
            className="metric-mono border border-accent/30 px-2 py-1 text-[11px] text-accent"
          >
            {activeHead}
          </span>
        ))}
        </div>
      ) : null}

      <Handle type="source" position={Position.Bottom} className="opacity-0" />
    </div>
  );
}
