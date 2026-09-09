import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import type { Message, TokenScope } from "../types";
import { DataViz } from "./DataViz";

interface Props {
  message: Message;
  onRate: (id: string, rating: "thumbs_up" | "thumbs_down") => void;
}

const CITATION_RE = /\[Source:\s*([^\]]+)\]/g;
/** Evidence markers the synthesizer resolved, e.g. [E3]. */
const MARKER_RE = /\[E(\d+)\]/g;
/** What the backend leaves behind where a citation did not resolve to any real
 *  piece of evidence. Plain text here reads like part of the sentence, which is
 *  exactly the wrong impression for the one thing the reader must not skim. */
const UNVERIFIED_RE = /\[unverified\]/g;

// What the reported number actually covers. An unqualified count invites the
// reader to assume it is the whole run, which is only true for "job".
/** Rendered inline after the count. A tooltip is not enough: it is invisible on
 *  touch devices, so every scope read as an unqualified whole-run total. */
const SCOPE_SUFFIXES: Record<TokenScope, string> = {
  job: "",
  answer: " (this answer)",
  synthesizer_only: " (synthesis only)",
  partial: " (partial)",
  unavailable: "",
  unknown: " (scope unknown)",
};

const SCOPE_LABELS: Record<TokenScope, string> = {
  job: "Total tokens for the full research run",
  answer: "Tokens for this answer",
  synthesizer_only: "Final synthesis step only — excludes research steps",
  partial: "Partial count — some steps were not measured",
  unavailable: "The model did not report token usage",
  unknown: "Scope not reported by the server",
};

export function MessageBubble({ message, onRate }: Props) {
  const isUser = message.role === "user";

  // Turn citation markers into inline-code badges. Substitution into markdown,
  // rendered through rehypeSanitize — no raw HTML is introduced.
  const displayContent = !isUser
    ? message.content
        .replace(CITATION_RE, "**`📎 $1`**")
        .replace(UNVERIFIED_RE, "**`⚠️ unverified`**")
        .replace(MARKER_RE, "`📎 E$1`")
    : message.content;

  return (
    <div
      className={`flex animate-fade-in ${
        isUser ? "justify-end" : "items-start gap-3"
      }`}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-warm-sage flex items-center justify-center flex-shrink-0 mt-1">
          <span className="text-white text-xs font-semibold">RA</span>
        </div>
      )}
      <div className={`max-w-[720px] ${isUser ? "max-w-[85%]" : "flex-1 max-w-[720px]"}`}>
        <div
          className={`px-4 py-3 ${
            isUser
              ? "bg-warm-user rounded-xl rounded-br-sm text-warm-text"
              : "bg-warm-card shadow-sm rounded-xl rounded-bl-sm text-warm-text"
          }`}
        >
          {isUser ? (
            <p className="leading-relaxed text-[15px]">{message.content}</p>
          ) : (
            <>
              <div className="leading-relaxed text-[15px] prose prose-sm max-w-none prose-headings:text-warm-text prose-p:text-warm-text prose-strong:text-warm-text prose-a:text-warm-accent prose-code:bg-warm-sage/10 prose-code:text-warm-sage prose-code:text-xs prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:font-medium prose-code:before:content-none prose-code:after:content-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeSanitize]}
                >
                  {displayContent}
                </ReactMarkdown>
                {message.streaming && (
                  <span className="streaming-cursor" aria-hidden="true">
                    &#9610;
                  </span>
                )}
              </div>
              {message.unverifiedCitations && message.unverifiedCitations.length > 0 && (
                <p className="mt-3 text-[13px] text-warm-rose">
                  {message.unverifiedCitations.length === 1
                    ? "1 citation in this report matched no evidence that was actually retrieved. It is marked unverified above; treat the claim it supports as unsupported."
                    : `${message.unverifiedCitations.length} citations in this report matched no evidence that was actually retrieved. They are marked unverified above; treat the claims they support as unsupported.`}
                </p>
              )}
              {message.tableData && <DataViz data={message.tableData} />}
            </>
          )}
        </div>
        {!isUser && !message.streaming && (
          <div className="mt-1.5 flex items-center gap-1">
            <button
              className={`px-2 py-0.5 rounded-md text-sm transition-colors duration-150 ${
                message.rating === "thumbs_up"
                  ? "bg-warm-accent/10 border border-warm-accent/30"
                  : "border border-transparent hover:border-warm-border text-warm-text-secondary hover:text-warm-text"
              }`}
              onClick={() => onRate(message.id, "thumbs_up")}
              aria-label="Rate response as helpful"
              title="Helpful"
            >
              👍
            </button>
            <button
              className={`px-2 py-0.5 rounded-md text-sm transition-colors duration-150 ${
                message.rating === "thumbs_down"
                  ? "bg-warm-rose/10 border border-warm-rose/30"
                  : "border border-transparent hover:border-warm-border text-warm-text-secondary hover:text-warm-text"
              }`}
              onClick={() => onRate(message.id, "thumbs_down")}
              aria-label="Rate response as not helpful"
              title="Not helpful"
            >
              👎
            </button>
            {message.tokenUsage && message.tokenUsage.scope !== "unavailable" && (
              <span
                className="ml-auto text-[11px] text-warm-text-secondary tabular-nums"
                title={SCOPE_LABELS[message.tokenUsage.scope]}
              >
                {(message.tokenUsage.input + message.tokenUsage.output).toLocaleString()} tokens
                {SCOPE_SUFFIXES[message.tokenUsage.scope]}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
