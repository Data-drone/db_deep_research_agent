import type { JobStatus } from "../types";

const NODE_LABELS: Record<string, string> = {
  clarifier: "Clarifying query",
  planner: "Planning research",
  authorizer: "Checking permissions",
  researcher: "Gathering evidence",
  normalizer: "Normalizing data",
  evaluator: "Evaluating sufficiency",
  compressor: "Compressing findings",
  synthesizer: "Synthesizing answer",
  verifier: "Verifying citations",
};

const NODE_ORDER = [
  "clarifier", "planner", "authorizer", "researcher",
  "normalizer", "evaluator", "compressor", "synthesizer", "verifier",
];

interface Props {
  job: JobStatus;
  onCancel: () => void;
}

export function ProgressBar({ job, onCancel }: Props) {
  const currentNode = job.current_node;
  const currentIdx = currentNode ? NODE_ORDER.indexOf(currentNode) : -1;

  const label = currentNode
    ? NODE_LABELS[currentNode] || currentNode
    : job.status === "pending"
      ? "Starting..."
      : "Processing...";

  const progress =
    currentIdx >= 0 ? ((currentIdx + 1) / NODE_ORDER.length) * 100 : 10;

  return (
    <div className="my-3 p-3 bg-warm-card shadow-sm rounded-xl animate-fade-in" aria-busy="true">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-warm-text font-medium" role="status" aria-live="polite">{label}</span>
        <button className="text-warm-text-secondary hover:text-warm-rose transition-colors duration-150 text-sm px-1" onClick={onCancel} aria-label="Cancel research" title="Cancel">✕</button>
      </div>
      <div className="h-1 bg-warm-border rounded-full overflow-hidden" role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100} aria-label={`Research progress: ${label}`}>
        <div className="h-full bg-warm-accent rounded-full transition-all duration-300 ease-out" style={{ width: `${progress}%` }} />
      </div>
      <div className="flex justify-between mt-1.5" aria-hidden="true">
        {NODE_ORDER.map((node, idx) => (
          <span key={node} className={`text-[10px] transition-colors duration-150 ${idx < currentIdx ? "text-warm-accent" : idx === currentIdx ? "text-warm-text" : "text-warm-border"}`} title={NODE_LABELS[node]}>•</span>
        ))}
      </div>
    </div>
  );
}
