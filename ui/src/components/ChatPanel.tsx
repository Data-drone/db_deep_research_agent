import { useState, useRef, useEffect } from "react";
import { MessageBubble } from "./MessageBubble";
import { ProgressBar } from "./ProgressBar";
import type { Message, JobStatus, OutputMode, ResponseMode } from "../types";

interface Props {
  messages: Message[];
  currentJob: JobStatus | null;
  isLoading: boolean;
  selectedTools: string[];
  outputMode: OutputMode;
  responseMode: ResponseMode;
  onSend: (query: string, tools: string[], mode: OutputMode, responseMode: ResponseMode) => void;
  onCancel: () => void;
  onRate: (id: string, rating: "thumbs_up" | "thumbs_down") => void;
}

export function ChatPanel({
  messages,
  currentJob,
  isLoading,
  selectedTools,
  outputMode,
  responseMode,
  onSend,
  onCancel,
  onRate,
}: Props) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const sampleQuestions = [
    "What was CBA's closing share price last Friday?",
    "Compare the FY2024 revenue of the top 4 ASX banks",
    "What are the key risk factors mentioned in ANZ's annual report?",
    "Summarise the latest RBA interest rate decision and its impact on equities",
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentJob]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed, selectedTools, outputMode, responseMode);
    setInput("");
  };

  return (
    <div className="chat-panel">
      <div className="messages-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <h3>Research and Chat Agent</h3>
            <p>Ask a question to start researching, or try one of these:</p>
            <div className="sample-questions">
              {sampleQuestions.map((q) => (
                <button
                  key={q}
                  className="sample-question"
                  onClick={() => onSend(q, selectedTools, outputMode, responseMode)}
                  disabled={isLoading}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onRate={onRate} />
        ))}
        {currentJob && <ProgressBar job={currentJob} onCancel={onCancel} />}
        <div ref={messagesEndRef} />
      </div>
      <form className="input-area" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a research question..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
