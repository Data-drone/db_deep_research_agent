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
          <div
            className="markdown-content"
            dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
          />
        )}
      </div>
      {!isUser && (
        <div className="message-actions">
          <button
            className={`rate-btn ${message.rating === "thumbs_up" ? "active" : ""}`}
            onClick={() => onRate(message.id, "thumbs_up")}
            title="Helpful"
          >
            👍
          </button>
          <button
            className={`rate-btn ${message.rating === "thumbs_down" ? "active" : ""}`}
            onClick={() => onRate(message.id, "thumbs_down")}
            title="Not helpful"
          >
            👎
          </button>
        </div>
      )}
    </div>
  );
}

function formatContent(content: string): string {
  return content
    .replace(/^## (.+)$/gm, "<h3>$1</h3>")
    .replace(/^### (.+)$/gm, "<h4>$1</h4>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[Source: (.+?)\]/g, '<span class="citation">[Source: $1]</span>')
    .replace(/\n/g, "<br />");
}
