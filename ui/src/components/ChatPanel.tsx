import { useState, useRef, useEffect } from "react";
import { MessageBubble } from "./MessageBubble";
import { ProgressBar } from "./ProgressBar";
import type { Message, JobStatus, OutputMode } from "../types";

interface Props {
  messages: Message[];
  currentJob: JobStatus | null;
  isLoading: boolean;
  selectedTools: string[];
  outputMode: OutputMode;
  onSend: (query: string, tools: string[], mode: OutputMode) => void;
  onCancel: () => void;
  onRate: (id: string, rating: "thumbs_up" | "thumbs_down") => void;
}

export function ChatPanel({
  messages,
  currentJob,
  isLoading,
  selectedTools,
  outputMode,
  onSend,
  onCancel,
  onRate,
}: Props) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentJob]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed, selectedTools, outputMode);
    setInput("");
  };

  return (
    <div className="chat-panel">
      <div className="messages-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <h3>Deep Research Agent</h3>
            <p>Ask a question to start researching.</p>
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
