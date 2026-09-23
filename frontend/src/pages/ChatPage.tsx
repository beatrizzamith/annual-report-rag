import { useEffect, useRef, useState } from "react";

import { ApiError, clearMessages, listMessages, sendChatMessage } from "../api";
import Badge, { type BadgeTone } from "../components/Badge";
import CitationCard from "../components/CitationCard";
import { groupCitations, parseCitations } from "../citations";
import type { ChatMessage, ChatResponse, Citation } from "../types";

/** One rendered chat turn: either a stored message or a freshly answered one with its live status. */
interface DisplayTurn {
  role: "user" | "assistant";
  content: string;
  interpretedAs?: string | null;
  citations?: Citation[];
  status?: ChatResponse["status"];
  caveats?: string[];
}

const STATUS_TONE: Record<NonNullable<DisplayTurn["status"]>, BadgeTone> = {
  answered: "success",
  partial: "warning",
  not_found: "neutral",
};

const STATUS_LABEL: Record<NonNullable<DisplayTurn["status"]>, string> = {
  answered: "Answered",
  partial: "Partial",
  not_found: "Not found",
};

/** Small numbered citation marker, inline with the answer prose. */
function CitationMarker({ number }: { number: number }) {
  return (
    <sup className="inline-flex items-center justify-center h-3.5 w-3.5 mx-0.5 rounded-full bg-brand-100 text-brand-700 text-[9px] font-semibold not-italic align-super">
      {number}
    </sup>
  );
}

/** Small circular avatar with the role's initial. */
function Avatar({ role }: { role: "user" | "assistant" }) {
  const isUser = role === "user";
  return (
    <div
      className={`h-7 w-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 ${
        isUser ? "bg-gray-700 text-white" : "bg-gradient-to-br from-brand-500 to-brand-700 text-white"
      }`}
    >
      {isUser ? "U" : "A"}
    </div>
  );
}

/**
 * Converts persisted chat messages into display turns.
 *
 * `interpreted_as` is stored on the assistant row (the rewrite happens
 * before the answer call), but it is shown under the preceding *user*
 * bubble ("Interpreted as: ..."), matching where a live answer attaches
 * it. This walks the messages once, patching each user turn from the
 * assistant reply that follows it.
 *
 * @param messages - Persisted chat messages, oldest first.
 * @returns Display turns in the same order, with `interpretedAs` attached
 *   to the correct (user) turn.
 */
function messagesToTurns(messages: ChatMessage[]): DisplayTurn[] {
  const turns: DisplayTurn[] = [];
  for (const message of messages) {
    if (message.role === "assistant" && message.interpreted_as) {
      const previousTurn = turns[turns.length - 1];
      if (previousTurn?.role === "user") {
        previousTurn.interpretedAs = message.interpreted_as;
      }
    }
    turns.push({
      role: message.role,
      content: message.content,
      citations: message.citations ?? undefined,
    });
  }
  return turns;
}

/**
 * Renders one chat turn (user bubble, or assistant answer with evidence).
 *
 * Evidence cards default open only for the latest turn (`isLatest`), so a
 * question you just asked shows its sources immediately without a click,
 * while older turns start collapsed — otherwise a long conversation piles
 * up every past answer's evidence cards and pushes recent messages, and
 * the input box, out of view.
 */
function TurnView({ turn, isLatest }: { turn: DisplayTurn; isLatest: boolean }) {
  const [expanded, setExpanded] = useState(isLatest);

  if (turn.role === "user") {
    return (
      <div className="flex items-start gap-2.5 justify-end">
        <div className="max-w-lg">
          <div className="bg-brand-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm shadow-sm">
            {turn.content}
          </div>
          {turn.interpretedAs && (
            <div className="text-xs text-gray-400 mt-1 text-right">
              Interpreted as: {turn.interpretedAs}
            </div>
          )}
        </div>
        <Avatar role="user" />
      </div>
    );
  }

  const { segments, numberBySourceId } = parseCitations(turn.content, turn.citations ?? []);
  const groups = groupCitations(turn.citations ?? [], numberBySourceId);

  return (
    <div className="flex items-start gap-2.5">
      <Avatar role="assistant" />
      <div className="max-w-2xl space-y-2.5 flex-1">
        <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-card space-y-2.5">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-gray-500">Assistant</span>
            {turn.status && <Badge tone={STATUS_TONE[turn.status]}>{STATUS_LABEL[turn.status]}</Badge>}
          </div>
          <p className="text-sm text-gray-800 leading-relaxed">
            {segments.map((segment, index) =>
              segment.citationNumber !== null ? (
                <CitationMarker key={index} number={segment.citationNumber} />
              ) : (
                <span key={index}>{segment.text}</span>
              ),
            )}
          </p>
          {turn.caveats && turn.caveats.length > 0 && (
            <p className="text-xs text-amber-700 bg-amber-50 rounded-md px-2.5 py-1.5">
              {turn.caveats.join(" · ")}
            </p>
          )}
        </div>
        {groups.length > 0 && (
          <>
            <button
              onClick={() => setExpanded((previous) => !previous)}
              className="text-xs font-medium text-gray-400 hover:text-gray-600 transition-colors flex items-center gap-1"
            >
              <span className={`transition-transform ${expanded ? "rotate-90" : ""}`}>›</span>
              {expanded ? "Hide" : "Show"} {groups.length} source{groups.length === 1 ? "" : "s"}
            </button>
            {expanded && (
              <div className="space-y-2">
                {groups.map((group, index) => (
                  <CitationCard key={index} group={group} number={group.number} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** Empty state prompting a first question. */
function EmptyState() {
  return (
    <div className="text-center py-16 text-gray-400">
      <div className="text-3xl mb-2">💬</div>
      <p className="text-sm">
        Ask a question about the uploaded reports, e.g. &ldquo;How many FTE does Shell
        have?&rdquo;
      </p>
    </div>
  );
}

/** Chat page: ask questions and review cited answers with verbatim evidence. */
export default function ChatPage() {
  const [turns, setTurns] = useState<DisplayTurn[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listMessages().then((messages) => setTurns(messagesToTurns(messages)));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, sending]);

  async function handleSend(event: React.FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || sending) {
      return;
    }
    setInput("");
    setError(null);
    setTurns((previous) => [...previous, { role: "user", content: text }]);
    setSending(true);
    try {
      const response = await sendChatMessage(text);
      setTurns((previous) => [
        ...previous.map((turn, index) =>
          index === previous.length - 1 ? { ...turn, interpretedAs: response.interpreted_as } : turn,
        ),
        {
          role: "assistant",
          content: response.answer,
          citations: response.citations,
          status: response.status,
          caveats: response.caveats,
        },
      ]);
    } catch (err) {
      if (err instanceof ApiError && err.errorCode === "no_reports_loaded") {
        setTurns((previous) => [
          ...previous,
          {
            role: "assistant",
            content: "No reports have been uploaded yet. Go to the Reports tab to upload one.",
          },
        ]);
      } else {
        setError(err instanceof ApiError ? err.message : "Something went wrong.");
      }
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      formRef.current?.requestSubmit();
    }
  }

  async function handleNewChat() {
    if (turns.length === 0) {
      return;
    }
    const confirmed = window.confirm("Start a new chat? This clears the current conversation.");
    if (!confirmed) {
      return;
    }
    await clearMessages();
    setTurns([]);
    setError(null);
  }

  return (
    <div className="flex flex-col h-[calc(100vh-9.5rem)]">
      <div className="flex justify-end mb-2">
        <button
          onClick={handleNewChat}
          className="text-xs font-medium text-gray-400 hover:text-gray-600 transition-colors"
        >
          + New chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto space-y-4 pb-4 pr-1">
        {turns.length === 0 && <EmptyState />}
        {turns.map((turn, index) => (
          <TurnView key={index} turn={turn} isLatest={index === turns.length - 1} />
        ))}
        {sending && (
          <div className="flex items-center gap-2.5 text-sm text-gray-400">
            <Avatar role="assistant" />
            <span className="inline-flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-gray-300 animate-bounce [animation-delay:-0.3s]" />
              <span className="h-1.5 w-1.5 rounded-full bg-gray-300 animate-bounce [animation-delay:-0.15s]" />
              <span className="h-1.5 w-1.5 rounded-full bg-gray-300 animate-bounce" />
              <span className="ml-1">Searching and verifying…</span>
            </span>
          </div>
        )}
        {error && <p className="text-sm text-rose-600">{error}</p>}
        <div ref={bottomRef} />
      </div>

      <form
        ref={formRef}
        onSubmit={handleSend}
        className="flex gap-2 border-t border-gray-100 pt-3 bg-gray-50"
      >
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder="Ask a question about the reports..."
          className="flex-1 border border-gray-200 rounded-xl px-4 py-2.5 text-sm resize-none outline-none transition-shadow focus:ring-2 focus:ring-brand-500/30 focus:border-brand-400 bg-white"
        />
        <button
          type="submit"
          disabled={sending || !input.trim()}
          className="bg-brand-600 hover:bg-brand-700 text-white rounded-xl px-5 py-2.5 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </form>
    </div>
  );
}
