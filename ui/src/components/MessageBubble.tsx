import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import type { Message } from "../types";

interface Props {
  message: Message;
  onRate: (id: string, rating: "thumbs_up" | "thumbs_down") => void;
}

export function MessageBubble({ message, onRate }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="message-content">
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div className="markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeSanitize]}
            >
              {message.content}
            </ReactMarkdown>
            {message.streaming && (
              <span className="streaming-cursor" aria-hidden="true">&#9610;</span>
            )}
          </div>
        )}
      </div>
      {!isUser && (
        <div className="message-actions">
          <button
            className={`rate-btn ${message.rating === "thumbs_up" ? "active" : ""}`}
            onClick={() => onRate(message.id, "thumbs_up")}
            aria-label="Rate response as helpful"
            title="Helpful"
          >
            👍
          </button>
          <button
            className={`rate-btn ${message.rating === "thumbs_down" ? "active" : ""}`}
            onClick={() => onRate(message.id, "thumbs_down")}
            aria-label="Rate response as not helpful"
            title="Not helpful"
          >
            👎
          </button>
        </div>
      )}
    </div>
  );
}
