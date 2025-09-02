interface ProgressBarProps {
  value: number;
}

export default function ProgressBar({ value }: ProgressBarProps) {
  return (
    <div className="h-2 w-full bg-muted">
      <div className="h-2 bg-primary" style={{ width: `${value}%` }} />
    </div>
  );
}

