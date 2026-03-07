import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { ToolSelector } from "./components/ToolSelector";
import { ReportPanel } from "./components/ReportPanel";
import { useResearch } from "./hooks/useResearch";
import type { OutputMode } from "./types";
import "./App.css";

function App() {
  const { messages, currentJob, isLoading, send, cancel, rate } =
    useResearch();
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [outputMode, setOutputMode] = useState<OutputMode>("chat");

  const lastAssistantMsg = [...messages]
    .reverse()
    .find((m) => m.role === "assistant");
  const showReport = outputMode === "report" && lastAssistantMsg;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Deep Research Agent</h1>
      </header>
      <div className="app-body">
        <aside className="sidebar">
          <ToolSelector
            selectedTools={selectedTools}
            onToolsChange={setSelectedTools}
            outputMode={outputMode}
            onOutputModeChange={setOutputMode}
          />
        </aside>
        <main className="main-content">
          {showReport ? (
            <ReportPanel content={lastAssistantMsg.content} />
          ) : (
            <ChatPanel
              messages={messages}
              currentJob={currentJob}
              isLoading={isLoading}
              selectedTools={selectedTools}
              outputMode={outputMode}
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
