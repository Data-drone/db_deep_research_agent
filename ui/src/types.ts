export interface Tool {
  name: string;
  display_name: string;
  risk_tier: "safe" | "restricted" | "privileged";
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  jobId?: string;
  rating?: "thumbs_up" | "thumbs_down";
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "cancelled" | "failed";
  result?: string;
  current_node?: string;
}

export type OutputMode = "chat" | "report";
