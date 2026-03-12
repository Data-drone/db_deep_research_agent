import { useEffect, useState } from "react";
import { fetchTools } from "../api";
import type { OutputMode, ResponseMode, Tool } from "../types";

interface Props {
  selectedTools: string[];
  onToolsChange: (tools: string[]) => void;
  outputMode: OutputMode;
  onOutputModeChange: (mode: OutputMode) => void;
  responseMode: ResponseMode;
  onResponseModeChange: (mode: ResponseMode) => void;
}

const RISK_STYLES: Record<string, string> = {
  safe: "bg-warm-sage/15 text-warm-sage",
  restricted: "bg-warm-amber/15 text-warm-amber",
  privileged: "bg-warm-rose/15 text-warm-rose",
};

export function ToolSelector({ selectedTools, onToolsChange, outputMode, onOutputModeChange, responseMode, onResponseModeChange }: Props) {
  const [tools, setTools] = useState<Tool[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTools().then(setTools).catch(() => setError("Failed to load tools"));
  }, []);

  const toggle = (name: string) => {
    if (selectedTools.includes(name)) {
      onToolsChange(selectedTools.filter((t) => t !== name));
    } else {
      onToolsChange([...selectedTools, name]);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-warm-text-secondary mb-3">Tools</h3>
        {error && <p className="text-sm text-warm-rose">{error}</p>}
        {tools.length === 0 && !error && <p className="text-sm text-warm-text-secondary">No tools available</p>}
        <ul className="flex flex-col gap-2">
          {tools.map((tool) => (
            <li key={tool.name}>
              <label className="flex items-center gap-2.5 cursor-pointer text-sm group">
                <input type="checkbox" checked={selectedTools.includes(tool.name)} onChange={() => toggle(tool.name)} className="w-4 h-4 rounded border-warm-border text-warm-accent focus:ring-warm-accent/30 accent-warm-accent" />
                <span className="flex-1 text-warm-text group-hover:text-warm-accent transition-colors duration-150">{tool.display_name}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium uppercase ${RISK_STYLES[tool.risk_tier] || ""}`}>{tool.risk_tier}</span>
              </label>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-warm-text-secondary mb-2">Response Mode</h3>
        <div className="flex flex-col gap-1.5">
          {(["quick", "research"] as const).map((mode) => (
            <label key={mode} className="flex items-center gap-2.5 cursor-pointer text-sm">
              <input type="radio" name="responseMode" value={mode} checked={responseMode === mode} onChange={() => onResponseModeChange(mode)} className="w-4 h-4 border-warm-border text-warm-accent focus:ring-warm-accent/30 accent-warm-accent" />
              <span className="text-warm-text">{mode === "quick" ? "Quick Reply" : "Deep Research"}</span>
            </label>
          ))}
        </div>
      </div>
      {responseMode === "research" && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-warm-text-secondary mb-2">Output Format</h3>
          <div className="flex flex-col gap-1.5">
            {(["chat", "report"] as const).map((mode) => (
              <label key={mode} className="flex items-center gap-2.5 cursor-pointer text-sm">
                <input type="radio" name="outputMode" value={mode} checked={outputMode === mode} onChange={() => onOutputModeChange(mode)} className="w-4 h-4 border-warm-border text-warm-accent focus:ring-warm-accent/30 accent-warm-accent" />
                <span className="text-warm-text capitalize">{mode}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
