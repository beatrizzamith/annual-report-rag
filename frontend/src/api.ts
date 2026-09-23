import type {
  ApiErrorBody,
  ChatMessage,
  ChatResponse,
  ExtractionItem,
  HealthStatus,
  ReportSummary,
} from "./types";

/** Thrown when the backend returns a typed error payload. */
export class ApiError extends Error {
  errorCode: string;

  constructor(body: ApiErrorBody) {
    super(body.message);
    this.errorCode = body.error_code;
  }
}

/**
 * Sends a request to the backend and parses its JSON body.
 *
 * @param path - API path, e.g. "/api/reports".
 * @param init - Standard `fetch` options.
 * @returns The parsed JSON response body.
 * @throws ApiError if the response is not ok and carries a typed error body.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const body = await response.json();
  if (!response.ok) {
    throw new ApiError(body as ApiErrorBody);
  }
  return body as T;
}

export function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/api/health");
}

export function listReports(): Promise<ReportSummary[]> {
  return request<ReportSummary[]>("/api/reports");
}

/**
 * Uploads a PDF for ingestion.
 *
 * @param file - The PDF file to upload.
 * @param company - The company name typed at upload.
 * @param fiscalYear - The fiscal year typed at upload.
 * @returns The created or existing report.
 */
export function uploadReport(
  file: File,
  company: string,
  fiscalYear: number,
): Promise<ReportSummary> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("company", company);
  formData.append("fiscal_year", String(fiscalYear));
  return request<ReportSummary>("/api/reports", { method: "POST", body: formData });
}

export function getExtractions(reportId: number): Promise<ExtractionItem[]> {
  return request<ExtractionItem[]>(`/api/reports/${reportId}/extractions`);
}

/**
 * Permanently deletes a report, so its file can be re-uploaded and
 * re-ingested from scratch.
 *
 * @param reportId - The report to delete.
 */
export function deleteReport(reportId: number): Promise<{ deleted: boolean }> {
  return request<{ deleted: boolean }>(`/api/reports/${reportId}`, { method: "DELETE" });
}

export function sendChatMessage(message: string): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
}

export function listMessages(): Promise<ChatMessage[]> {
  return request<ChatMessage[]>("/api/messages");
}

/** Clears the chat history, starting a fresh conversation. */
export function clearMessages(): Promise<{ cleared: boolean }> {
  return request<{ cleared: boolean }>("/api/messages", { method: "DELETE" });
}
