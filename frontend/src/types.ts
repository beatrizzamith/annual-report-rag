/** A report row as returned by GET /api/reports and POST /api/reports. */
export interface ReportSummary {
  id: number;
  filename: string;
  company: string;
  fiscal_year: number;
  page_count: number | null;
  status: "processing" | "ready" | "failed";
  stage: string | null;
  progress: number;
  error: string | null;
  created_at: string;
  extraction_counts: { fte: number; sustainability_goal: number };
}

/** One pre-extracted item as returned by GET /api/reports/{id}/extractions. */
export interface ExtractionItem {
  id: number;
  kind: "fte" | "sustainability_goal";
  payload: FtePayload | GoalPayload;
  quote: string;
  page: number;
  verified: boolean;
}

export interface FtePayload {
  found: boolean;
  value_text: string | null;
  value?: number | null;
  metric: "fte" | "headcount" | "average_fte" | "other";
  as_of: string | null;
  scope: string | null;
  notes: string | null;
}

export interface GoalPayload {
  title: string;
  category: string;
  target: string | null;
  target_year: number | null;
  baseline: string | null;
}

/**
 * One verified citation backing an inline [S3]-style marker in a chat
 * answer. `label`/`value_text`/`unit`/`period` are null when the citation
 * backs non-numeric prose rather than a specific figure — every inline
 * marker has one of these, so every citation can be inspected.
 */
export interface Citation {
  label: string | null;
  value_text: string | null;
  unit: string | null;
  period: string | null;
  quote: string;
  report: string;
  page: string;
  verified: boolean;
  // Matches the "[S3]"-style marker cited inline in the answer text.
  // Optional: chat history persisted before this field was added lacks it.
  source_id?: string;
  // The cited chunk's full text, for "show full source" in the UI.
  // Optional: chat history persisted before this field was added lacks it.
  chunk_text?: string;
}

/** The full response body of POST /api/chat. */
export interface ChatResponse {
  status: "answered" | "partial" | "not_found";
  answer: string;
  citations: Citation[];
  caveats: string[];
  interpreted_as: string | null;
}

/** One stored chat message as returned by GET /api/messages. */
export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  interpreted_as: string | null;
  created_at: string;
}

/** The typed error body every failing API call returns. */
export interface ApiErrorBody {
  error_code: string;
  message: string;
}

export interface HealthStatus {
  status: string;
  llm_configured: boolean;
  llm_provider: string;
  fts5_available: boolean;
  chunks_indexed: number;
}
