import { useEffect, useState } from "react";
import { api } from "./api";

const cache = new Map<string, unknown>();

export function useApi<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const cached = cache.get(path);
    if (cached) {
      setData(cached as T);
      setLoading(false);
      return;
    }
    setLoading(true);
    api<T>(path)
      .then((d) => {
        if (!cancelled) {
          cache.set(path, d);
          setData(d);
        }
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return { data, error, loading };
}

