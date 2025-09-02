import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

interface Toast {
  id: number;
  message: string;
}

const ToastContext = createContext<(msg: string) => void>(() => {});

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const push = (message: string) => {
    const t = { id: Date.now(), message };
    setToasts((cur) => [...cur, t]);
    setTimeout(
      () => setToasts((cur) => cur.filter((x) => x.id !== t.id)),
      3000
    );
  };

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="fixed bottom-4 right-4 space-y-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="rounded bg-gray-800 px-4 py-2 text-white"
          >
            {t.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

