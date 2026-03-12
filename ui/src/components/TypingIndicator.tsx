export function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-warm-sage flex items-center justify-center flex-shrink-0">
        <span className="text-white text-xs font-semibold">RA</span>
      </div>
      <div className="bg-warm-card shadow-sm rounded-xl rounded-bl-sm px-4 py-3">
        <div className="flex gap-1.5">
          <span
            className="w-2 h-2 rounded-full bg-warm-accent-light animate-bounce-dot"
            style={{ animationDelay: "0ms" }}
          />
          <span
            className="w-2 h-2 rounded-full bg-warm-accent-light animate-bounce-dot"
            style={{ animationDelay: "200ms" }}
          />
          <span
            className="w-2 h-2 rounded-full bg-warm-accent-light animate-bounce-dot"
            style={{ animationDelay: "400ms" }}
          />
        </div>
      </div>
    </div>
  );
}
