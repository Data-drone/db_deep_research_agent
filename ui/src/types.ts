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
  tokenUsage?: { input: number; output: number; scope: "answer" | "job" };
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "cancelled" | "failed";
  result?: string;
  current_node?: string;
  error?: string;
}

export type OutputMode = "chat" | "report";

export type ResponseMode = "quick" | "research";

export interface ClarificationRequest {
  jobId: string;
  clarificationId: string;
  question: string;
  options: string[];
}
