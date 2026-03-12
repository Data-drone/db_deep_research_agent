import { useState, useRef, useEffect, useCallback } from "react";
import { MessageBubble } from "./MessageBubble";
import { ProgressBar } from "./ProgressBar";
import { TypingIndicator } from "./TypingIndicator";
import { ClarificationPrompt } from "./ClarificationPrompt";
import type { Message, JobStatus, OutputMode, ResponseMode, ClarificationRequest } from "../types";

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
  clarificationRequest: ClarificationRequest | null;
  clarificationSubmitting: boolean;
  onAnswerClarification: (answer: string) => void;
}

const sampleQuestions = [
  "What was CBA's closing share price last Friday?",
  "Compare the FY2024 revenue of the top 4 ASX banks",
  "What are the key risk factors mentioned in ANZ's annual report?",
  "Summarise the latest RBA interest rate decision and its impact on equities",
];

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
  clarificationRequest,
  clarificationSubmitting,
  onAnswerClarification,
}: Props) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentJob]);

  const autoResize = useCallback(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, []);

  useEffect(() => {
    autoResize();
  }, [input, autoResize]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed, selectedTools, outputMode, responseMode);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const showTypingIndicator =
    isLoading && !messages.some((m) => m.streaming);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto flex flex-col gap-4">
          {messages.length === 0 && (
            <div className="text-center mt-24 animate-fade-in">
              <h3 className="text-2xl font-semibold text-warm-text mb-2">
                What are you curious about today?
              </h3>
              <p className="text-warm-text-secondary mb-6">
                Ask a question to start researching, or try one of these:
              </p>
              <div className="flex flex-wrap gap-3 justify-center max-w-xl mx-auto">
                {sampleQuestions.map((q) => (
                  <button
                    key={q}
                    className="px-4 py-2.5 border border-warm-border rounded-xl bg-warm-card text-warm-text text-sm text-left shadow-sm transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 hover:border-warm-accent/40 disabled:opacity-40 disabled:cursor-default"
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
          {clarificationRequest && (
            <ClarificationPrompt
              request={clarificationRequest}
              onAnswer={onAnswerClarification}
              submitting={clarificationSubmitting}
            />
          )}
          {showTypingIndicator && <TypingIndicator />}
          {currentJob && <ProgressBar job={currentJob} onCancel={onCancel} />}
          <div ref={messagesEndRef} />
        </div>
      </div>
      <div className="border-t border-warm-border bg-warm-bg px-4 py-3">
        <form
          className="max-w-3xl mx-auto flex gap-3 items-end"
          onSubmit={handleSubmit}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything — I'll help you find the answer"
            disabled={isLoading}
            rows={1}
            className="flex-1 px-4 py-2.5 border border-warm-border rounded-xl bg-warm-input text-warm-text text-[15px] leading-relaxed resize-none placeholder:text-warm-placeholder focus:outline-none focus:border-warm-border-focus focus:ring-2 focus:ring-warm-accent/20 disabled:opacity-50 transition-colors duration-150"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-5 py-2.5 rounded-xl bg-warm-accent text-white font-medium text-[15px] transition-colors duration-150 hover:bg-warm-accent-hover disabled:opacity-40 disabled:cursor-default flex-shrink-0"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
