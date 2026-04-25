import type { Edge, Node } from "@xyflow/react";

import type { AttentionTrace, HeadMask, ModelTopology } from "@/lib/types";

export interface SynapseNodeData extends Record<string, unknown> {
  kind: "prompt" | "layer" | "response";
  label: string;
  subtitle: string;
  active: boolean;
  maskedCount: number;
  activeHeads: string[];
  activeScore?: string;
  layerIndex?: number;
}

export function buildSynapseGraph(
  topology: ModelTopology | null | undefined,
  trace: AttentionTrace | null | undefined,
  maskedHeads: HeadMask[],
  overlayTrace?: AttentionTrace | null | undefined,
): { nodes: Node<SynapseNodeData>[]; edges: Edge[] } {
  if (!topology) {
    return { nodes: [], edges: [] };
  }
  const latestStep = trace?.steps.at(-1);
  const overlayStep = overlayTrace?.steps.at(-1);
  const activeLayerNames = latestStep?.high_activation_path.map((item) => item.split(":")[0]) ?? [];
  const overlayLayerNames = overlayStep?.high_activation_path.map((item) => item.split(":")[0]) ?? [];
  const activeLayerSet = new Set(activeLayerNames);
  const overlayLayerSet = new Set(overlayLayerNames);
  const maskedByLayer = maskedHeads.reduce<Record<number, number>>((accumulator, mask) => {
    accumulator[mask.layer_index] = (accumulator[mask.layer_index] ?? 0) + 1;
    return accumulator;
  }, {});

  const nodes: Node<SynapseNodeData>[] = [
    {
      id: "prompt",
      type: "synapse",
      position: { x: 0, y: 0 },
      data: {
        kind: "prompt",
        label: "Prompt Ingress",
        subtitle: trace?.source_prompt ? truncate(trace.source_prompt, 64) : "Awaiting probe.",
        active: activeLayerNames.length > 0,
        maskedCount: 0,
        activeHeads: [],
      },
    },
  ];

  for (const [positionIndex, layer] of topology.layers.entries()) {
    const layerCapture = latestStep?.layers.find(
      (candidate) => candidate.layer_index === layer.layer_index,
    );

    nodes.push({
      id: `layer-${layer.layer_index}`,
      type: "synapse",
      position: {
        x: positionIndex % 2 === 0 ? -120 : 120,
        y: (positionIndex + 1) * 180,
      },
      data: {
        kind: "layer",
        label: layer.layer_name,
        subtitle: `${layer.head_count} heads • d${layer.head_dim}`,
        active: activeLayerSet.has(layer.layer_name),
        maskedCount: maskedByLayer[layer.layer_index] ?? 0,
        activeHeads:
          layerCapture?.top_heads.slice(0, 2).map((head) => {
            return `${head.head_name} ${head.max_attention_score.toFixed(2)}`;
          }) ?? [],
        activeScore: layerCapture?.top_heads[0]
          ? layerCapture.top_heads[0].max_attention_score.toFixed(3)
          : undefined,
        layerIndex: layer.layer_index,
      },
    });
  }

  nodes.push({
    id: "response",
    type: "synapse",
    position: {
      x: 0,
      y: (topology.layers.length + 1) * 180,
    },
    data: {
      kind: "response",
      label: "Response Egress",
      subtitle: trace?.generated_text ? truncate(trace.generated_text, 68) : "No output yet.",
      active: Boolean(trace?.generated_text),
      maskedCount: 0,
      activeHeads: [],
    },
  });

  const chain = ["prompt", ...topology.layers.map((layer) => `layer-${layer.layer_index}`), "response"];
  const edges: Edge[] = [];

  for (let index = 0; index < chain.length - 1; index += 1) {
    const sourceId = chain[index];
    const targetId = chain[index + 1];
    const sourceLayerName = topology.layers[index - 1]?.layer_name;
    const targetLayerName = topology.layers[index]?.layer_name;

    const isPromptEdge = sourceId === "prompt" && Boolean(targetLayerName);
    const isResponseEdge = targetId === "response" && Boolean(sourceLayerName);
    const isActive =
      (isPromptEdge && activeLayerNames[0] === targetLayerName) ||
      (isResponseEdge && activeLayerNames.at(-1) === sourceLayerName) ||
      (sourceLayerName !== undefined &&
        targetLayerName !== undefined &&
        activeLayerSet.has(sourceLayerName) &&
        activeLayerSet.has(targetLayerName));

    edges.push({
      id: `${sourceId}->${targetId}`,
      source: sourceId,
      target: targetId,
      type: "smoothstep",
      animated: isActive,
      label: edgeLabel(sourceLayerName, targetLayerName, latestStep?.high_activation_path ?? []),
      labelShowBg: isActive,
      labelBgPadding: [10, 5],
      labelBgBorderRadius: 3,
      labelBgStyle: {
        fill: isActive ? "rgba(16,17,19,0.96)" : "rgba(16,17,19,0.9)",
        stroke: isActive ? "rgba(57,255,20,0.4)" : "rgba(127,132,143,0.22)",
        strokeWidth: 1,
      },
      labelStyle: {
        fill: isActive ? "#d9ffd1" : "#a1a1aa",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "-0.02em",
      },
      zIndex: isActive ? 12 : 4,
      style: {
        stroke: isActive ? "#39ff14" : "#3b3f46",
        strokeWidth: isActive ? 2.6 : 1.6,
        filter: isActive ? "drop-shadow(0 0 9px rgba(57,255,20,0.44))" : "none",
      },
    });
    // If overlay trace exists, add a secondary overlay edge showing the ablated path
    const overlayIsActive =
      (sourceId === "prompt" && Boolean(overlayLayerNames[0] && overlayLayerNames[0] === targetLayerName)) ||
      (targetId === "response" && Boolean(overlayLayerNames.at(-1) && overlayLayerNames.at(-1) === sourceLayerName)) ||
      (sourceLayerName !== undefined &&
        targetLayerName !== undefined &&
        overlayLayerSet.has(sourceLayerName) &&
        overlayLayerSet.has(targetLayerName));

    if (overlayTrace && overlayIsActive) {
      edges.push({
        id: `${sourceId}->${targetId}-overlay`,
        source: sourceId,
        target: targetId,
        type: "smoothstep",
        animated: true,
        label: edgeLabel(sourceLayerName, targetLayerName, overlayStep?.high_activation_path ?? []),
        labelShowBg: true,
        labelBgPadding: [8, 4],
        labelBgBorderRadius: 3,
        labelBgStyle: {
          fill: "rgba(16,17,19,0.96)",
          stroke: "rgba(255,77,109,0.28)",
          strokeWidth: 1,
        },
        labelStyle: {
          fill: "#ffd1d1",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          fontWeight: 700,
        },
        zIndex: 20,
        style: {
          stroke: "#ff4d6d",
          strokeWidth: 2.2,
          strokeDasharray: "6 4",
          filter: "drop-shadow(0 0 8px rgba(255,77,109,0.18))",
        },
      });
    }
  }

  return { nodes, edges };
}

function edgeLabel(
  sourceLayerName: string | undefined,
  targetLayerName: string | undefined,
  highActivationPath: string[],
) {
  if (!sourceLayerName && targetLayerName) {
    const firstHead = highActivationPath[0]?.split(":")[1];
    return firstHead ? compactHeadName(firstHead) : "";
  }

  if (sourceLayerName && !targetLayerName) {
    const lastHead = highActivationPath.at(-1)?.split(":")[1];
    return lastHead ? compactHeadName(lastHead) : "";
  }

  if (!sourceLayerName || !targetLayerName) {
    return "";
  }

  const sourceHead = highActivationPath.find((item) => item.startsWith(`${sourceLayerName}:`));
  const targetHead = highActivationPath.find((item) => item.startsWith(`${targetLayerName}:`));

  if (!sourceHead || !targetHead) {
    return "";
  }

  return compactEdgeLabel(sourceHead.split(":")[1], targetHead.split(":")[1]);
}

function truncate(text: string, maxLength: number) {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1)}…`;
}

function compactEdgeLabel(sourceHead: string, targetHead: string) {
  return `${compactHeadName(sourceHead)} -> ${compactHeadName(targetHead)}`;
}

function compactHeadName(headName: string) {
  const match = headName.match(/(\d+)/);

  if (!match) {
    return headName;
  }

  return `H${match[1]}`;
}
