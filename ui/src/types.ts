export interface Tool {
  name: string;
  display_name: string;
  risk_tier: "safe" | "restricted" | "privileged";
}

export interface ChartConfig {
  type: "line" | "bar" | "none";
  x: string;
  y: string[];
  title?: string;
}

export interface TableData {
  columns: { name: string; type: string }[];
  rows: string[][];
  sql?: string;
  chart?: ChartConfig;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string; // ISO string for serialization safety
  jobId?: string;
  rating?: "thumbs_up" | "thumbs_down";
  streaming?: boolean; // true while tokens are still arriving
  tableData?: TableData;
  tokenUsage?: { input: number; output: number; scope: TokenScope };
  /** "notice" marks status copy the hook wrote itself (cancelled, failed, a
   *  clarification outcome). The Report tab must not pick one of these as "the
   *  report", which is what happened while it just took the last assistant
   *  message. */
  kind?: "report" | "notice";
  /** Citations in this report that resolved to no real evidence. The report
   *  already carries an inline [unverified] where each one stood; this is the
   *  count for a summary line, so the reader is not relying on spotting them. */
  unverifiedCitations?: string[];
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "cancelled" | "failed";
  result?: string;
  current_node?: string;
  error?: string;
  token_usage?: { input: number; output: number; scope?: string };
  pending_clarification?: {
    clarification_id: string;
    question: string;
    options?: string[];
  };
  /** Evidence markers the report cited that resolve to nothing. Present on a
   *  terminal status so a client that missed the live event still learns of
   *  them. */
  unverified_citations?: string[];
}

/** What a token count actually covers. Never silently widen a narrow scope. */
export type TokenScope =
  | "job"
  | "answer"
  | "synthesizer_only"
  | "partial"
  | "unavailable"
  | "unknown";

export type OutputMode = "chat" | "report";

export type ResponseMode = "quick" | "research";

export interface ClarificationRequest {
  jobId: string;
  clarificationId: string;
  question: string;
  options: string[];
}
