import { useState } from "react";
import type { ClarificationRequest } from "../types";

interface Props {
  request: ClarificationRequest;
  onAnswer: (answer: string) => void;
  submitting?: boolean;
}

export function ClarificationPrompt({ request, onAnswer, submitting }: Props) {
  const [customInput, setCustomInput] = useState("");

  return (
    <div className="flex items-start gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-warm-amber flex items-center justify-center flex-shrink-0 mt-1">
        <span className="text-white text-xs font-semibold">?</span>
      </div>
      <div className="flex-1 max-w-[720px]">
        <div className="px-4 py-3 bg-warm-card shadow-sm rounded-xl rounded-bl-sm border border-warm-amber/30">
          <p className="text-[15px] font-medium text-warm-text mb-3">
            {request.question}
          </p>
          <div className="flex flex-wrap gap-2 mb-3">
            {request.options.map((option, idx) => (
              <button
                key={`${idx}-${option}`}
                className="px-3 py-1.5 rounded-lg border border-warm-border bg-warm-bg text-warm-text text-sm transition-all duration-150 hover:border-warm-accent hover:bg-warm-accent/5 hover:shadow-sm disabled:opacity-40"
                onClick={() => onAnswer(option)}
                disabled={submitting}
              >
                {option}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={customInput}
              onChange={(e) => setCustomInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && customInput.trim() && !submitting) {
                  onAnswer(customInput.trim());
                }
              }}
              placeholder="Or type your own answer..."
              disabled={submitting}
              className="flex-1 px-3 py-1.5 border border-warm-border rounded-lg bg-warm-input text-warm-text text-sm placeholder:text-warm-placeholder focus:outline-none focus:border-warm-border-focus focus:ring-2 focus:ring-warm-accent/20 transition-colors duration-150 disabled:opacity-40"
            />
            <button
              className="px-3 py-1.5 rounded-lg bg-warm-accent text-white text-sm font-medium transition-colors duration-150 hover:bg-warm-accent-hover disabled:opacity-40"
              disabled={!customInput.trim() || submitting}
              onClick={() => {
                if (customInput.trim()) onAnswer(customInput.trim());
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
