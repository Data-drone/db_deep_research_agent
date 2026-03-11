export interface Tool {
  name: string;
  display_name: string;
  risk_tier: "safe" | "restricted" | "privileged";
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string; // ISO string for serialization safety
  jobId?: string;
  rating?: "thumbs_up" | "thumbs_down";
  streaming?: boolean; // true while tokens are still arriving
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "cancelled" | "failed";
  result?: string;
  current_node?: string;
  error?: string;
}

export type OutputMode = "chat" | "report";
