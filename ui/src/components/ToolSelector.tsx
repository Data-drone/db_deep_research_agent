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

export function ToolSelector({
  selectedTools,
  onToolsChange,
  outputMode,
  onOutputModeChange,
  responseMode,
  onResponseModeChange,
}: Props) {
  const [tools, setTools] = useState<Tool[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTools()
      .then(setTools)
      .catch(() => setError("Failed to load tools"));
  }, []);

  const toggle = (name: string) => {
    if (selectedTools.includes(name)) {
      onToolsChange(selectedTools.filter((t) => t !== name));
    } else {
      onToolsChange([...selectedTools, name]);
    }
  };

  return (
    <div className="tool-selector">
      <h3>Tools</h3>
      {error && <p className="error">{error}</p>}
      {tools.length === 0 && !error && (
        <p className="muted">No tools available</p>
      )}
      <ul className="tool-list">
        {tools.map((tool) => (
          <li key={tool.name} className="tool-item">
            <label>
              <input
                type="checkbox"
                checked={selectedTools.includes(tool.name)}
                onChange={() => toggle(tool.name)}
              />
              <span className="tool-name">{tool.display_name}</span>
              <span className={`risk-badge risk-${tool.risk_tier}`}>
                {tool.risk_tier}
              </span>
            </label>
          </li>
        ))}
      </ul>

      <div className="response-mode-selector">
        <h3>Response Mode</h3>
        <label>
          <input
            type="radio"
            name="responseMode"
            value="quick"
            checked={responseMode === "quick"}
            onChange={() => onResponseModeChange("quick")}
          />
          Quick Reply
        </label>
        <label>
          <input
            type="radio"
            name="responseMode"
            value="research"
            checked={responseMode === "research"}
            onChange={() => onResponseModeChange("research")}
          />
          Deep Research
        </label>
      </div>

      {responseMode === "research" && (
        <div className="output-mode-selector">
          <h3>Output Format</h3>
          <label>
            <input
              type="radio"
              name="outputMode"
              value="chat"
              checked={outputMode === "chat"}
              onChange={() => onOutputModeChange("chat")}
            />
            Chat
          </label>
          <label>
            <input
              type="radio"
              name="outputMode"
              value="report"
              checked={outputMode === "report"}
              onChange={() => onOutputModeChange("report")}
            />
            Report
          </label>
        </div>
      )}
    </div>
  );
}
