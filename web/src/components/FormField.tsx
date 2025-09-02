import type { ReactNode } from "react";

interface FormFieldProps {
  label: string;
  children: ReactNode;
}

export default function FormField({ label, children }: FormFieldProps) {
  return (
    <label className="grid gap-1 text-sm">
      <span>{label}</span>
      {children}
    </label>
  );
}

