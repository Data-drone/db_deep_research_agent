import { useState } from "react";

interface ConversationItem {
  id: string;
  title: string;
  date: string;
}

const PLACEHOLDER_CONVERSATIONS: ConversationItem[] = [
  { id: "1", title: "CBA share price analysis", date: "Today" },
  { id: "2", title: "ASX bank revenue comparison", date: "Yesterday" },
  { id: "3", title: "RBA interest rate impacts", date: "Mar 9" },
  { id: "4", title: "ANZ annual report risks", date: "Mar 7" },
];

export function ConversationHistory() {
  const [activeId, setActiveId] = useState<string | null>(null);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-warm-text-secondary">
          History
        </h3>
        <button
          className="text-xs text-warm-accent hover:text-warm-accent-hover transition-colors duration-150 font-medium"
          onClick={() => setActiveId(null)}
        >
          + New Chat
        </button>
      </div>
      {PLACEHOLDER_CONVERSATIONS.map((conv) => (
        <button
          key={conv.id}
          onClick={() => setActiveId(conv.id)}
          className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors duration-150 ${
            activeId === conv.id
              ? "bg-warm-accent/10 text-warm-accent"
              : "text-warm-text hover:bg-warm-border/50"
          }`}
        >
          <div className="truncate font-medium">{conv.title}</div>
          <div className="text-xs text-warm-text-secondary mt-0.5">{conv.date}</div>
        </button>
      ))}
    </div>
  );
}
