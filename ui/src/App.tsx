import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { ToolSelector } from "./components/ToolSelector";
import { ReportPanel } from "./components/ReportPanel";
import { useResearch } from "./hooks/useResearch";
import type { OutputMode, ResponseMode } from "./types";
import "./App.css";

function App() {
  const { messages, currentJob, isLoading, error, send, cancel, rate } =
    useResearch();
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [outputMode, setOutputMode] = useState<OutputMode>("chat");
  const [responseMode, setResponseMode] = useState<ResponseMode>("quick");
  const [viewMode, setViewMode] = useState<"chat" | "report">("chat");

  const lastAssistantMsg = [...messages]
    .reverse()
    .find((m) => m.role === "assistant");
  const hasReport = outputMode === "report" && lastAssistantMsg;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Research and Chat Agent</h1>
        {hasReport && (
          <div className="view-tabs">
            <button
              className={`tab ${viewMode === "chat" ? "active" : ""}`}
              onClick={() => setViewMode("chat")}
            >
              Conversation
            </button>
            <button
              className={`tab ${viewMode === "report" ? "active" : ""}`}
              onClick={() => setViewMode("report")}
            >
              Report
            </button>
          </div>
        )}
      </header>
      <div className="app-body">
        <aside className="sidebar">
          <ToolSelector
            selectedTools={selectedTools}
            onToolsChange={setSelectedTools}
            outputMode={outputMode}
            onOutputModeChange={setOutputMode}
            responseMode={responseMode}
            onResponseModeChange={setResponseMode}
          />
        </aside>
        <main className="main-content">
          {error && (
            <div className="error-banner" role="alert">
              {error}
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
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
