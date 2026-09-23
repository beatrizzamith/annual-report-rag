import { useEffect, useState } from "react";

import { getHealth } from "../api";
import type { HealthStatus } from "../types";

/**
 * Loads the backend's health status once, for the "LLM key missing" banner.
 *
 * @returns The health status, or null while it is still loading or failed.
 */
export function useHealth(): HealthStatus | null {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return health;
}
