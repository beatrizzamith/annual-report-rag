import { useState } from "react";

import { useHealth } from "./hooks/useHealth";
import ChatPage from "./pages/ChatPage";
import ReportsPage from "./pages/ReportsPage";

type Page = "chat" | "reports";

/** Top-level nav tab, styled active/inactive within a segmented control. */
function NavTab({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3.5 py-1.5 rounded-md text-sm font-medium transition-all ${
        active
          ? "bg-white text-gray-900 shadow-sm"
          : "text-gray-500 hover:text-gray-800"
      }`}
    >
      {label}
    </button>
  );
}

/** Small mark used as the app's logo, in lieu of a real brand asset. */
function LogoMark() {
  return (
    <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-white font-semibold text-sm shrink-0">
      AR
    </div>
  );
}

/** Banner shown when the backend has no LLM provider configured. */
function MissingKeyBanner() {
  return (
    <div className="bg-amber-50 border-b border-amber-200/80 text-amber-800 text-sm px-4 py-2.5 text-center">
      No LLM provider is configured on the backend. Add an API key to{" "}
      <code className="bg-amber-100/80 rounded px-1 py-0.5 text-xs">.env</code> and restart the
      backend to enable chat and pre-extraction.
    </div>
  );
}

/** App shell: header, nav between Chat/Reports, and the missing-LLM-key banner. */
export default function App() {
  const [page, setPage] = useState<Page>("reports");
  const health = useHealth();

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <header className="border-b bg-white/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-5xl mx-auto flex items-center justify-between gap-4 px-4 sm:px-6 py-3">
          <div className="flex items-center gap-3">
            <LogoMark />
            <div className="leading-tight">
              <div className="font-semibold text-gray-900 text-sm">Annual Report Assistant</div>
              <div className="text-xs text-gray-400">Grounded Q&amp;A over annual reports</div>
            </div>
          </div>
          <nav className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            <NavTab label="Chat" active={page === "chat"} onClick={() => setPage("chat")} />
            <NavTab label="Reports" active={page === "reports"} onClick={() => setPage("reports")} />
          </nav>
        </div>
      </header>

      {health && !health.llm_configured && <MissingKeyBanner />}

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-6">
        {page === "chat" ? <ChatPage /> : <ReportsPage />}
      </main>
    </div>
  );
}
