import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { ToolSelector } from "./components/ToolSelector";
import { ReportPanel } from "./components/ReportPanel";
import { ConversationHistory } from "./components/ConversationHistory";
import { useResearch } from "./hooks/useResearch";
import type { OutputMode, ResponseMode } from "./types";

function App() {
  const { messages, currentJob, isLoading, error, clarificationRequest, clarificationSubmitting, send, cancel, answerClarification, rate } =
    useResearch();
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [outputMode, setOutputMode] = useState<OutputMode>("chat");
  const [responseMode, setResponseMode] = useState<ResponseMode>("quick");
  const [viewMode, setViewMode] = useState<"chat" | "report">("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const lastAssistantMsg = [...messages]
    .reverse()
    .find((m) => m.role === "assistant");
  const hasReport = outputMode === "report" && lastAssistantMsg;

  return (
    <div className="flex flex-col h-screen bg-warm-bg text-warm-text">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 py-3 border-b border-warm-border bg-warm-card">
        <button
          className="p-1.5 rounded-lg hover:bg-warm-sidebar transition-colors duration-150 text-warm-text-secondary lg:hidden"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          aria-label="Toggle sidebar"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>
        <h1 className="text-lg font-semibold text-warm-text">
          Research Assistant
        </h1>
        {hasReport && (
          <div className="flex gap-1 ml-auto">
            <button
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors duration-150 ${
                viewMode === "chat"
                  ? "bg-warm-accent text-white"
                  : "text-warm-text-secondary hover:bg-warm-sidebar"
              }`}
              onClick={() => setViewMode("chat")}
            >
              Conversation
            </button>
            <button
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors duration-150 ${
                viewMode === "report"
                  ? "bg-warm-accent text-white"
                  : "text-warm-text-secondary hover:bg-warm-sidebar"
              }`}
              onClick={() => setViewMode("report")}
            >
              Report
            </button>
          </div>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside
          className={`${
            sidebarOpen ? "w-[280px]" : "w-0"
          } border-r border-warm-border bg-warm-sidebar overflow-y-auto overflow-x-hidden transition-all duration-200 flex-shrink-0 flex flex-col`}
        >
          <div className="p-4 flex flex-col gap-6 min-w-[280px]">
            <ConversationHistory />
            <div className="border-t border-warm-border pt-4">
              <ToolSelector
                selectedTools={selectedTools}
                onToolsChange={setSelectedTools}
                outputMode={outputMode}
                onOutputModeChange={setOutputMode}
                responseMode={responseMode}
                onResponseModeChange={setResponseMode}
              />
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {error && (
            <div
              className="px-4 py-2.5 bg-warm-rose/10 text-warm-rose text-sm border-b border-warm-rose/20"
              role="alert"
            >
              Something went wrong. Please try again.
            </div>
          )}
          {hasReport && viewMode === "report" ? (
            <ReportPanel content={lastAssistantMsg.content} />
          ) : (
            <ChatPanel
              messages={messages}
              currentJob={currentJob}
              isLoading={isLoading}
              selectedTools={selectedTools}
              outputMode={outputMode}
              responseMode={responseMode}
              onSend={send}
              onCancel={cancel}
              onRate={rate}
              clarificationRequest={clarificationRequest}
              clarificationSubmitting={clarificationSubmitting}
              onAnswerClarification={answerClarification}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
