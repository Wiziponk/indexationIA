import { useCallback } from "react";

interface FileDropProps {
  onFiles: (files: FileList) => void;
}

export default function FileDrop({ onFiles }: FileDropProps) {
  const handle = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (e.dataTransfer.files) {
        onFiles(e.dataTransfer.files);
      }
    },
    [onFiles]
  );

  return (
    <div
      onDrop={handle}
      onDragOver={(e) => e.preventDefault()}
      className="flex items-center justify-center rounded border-2 border-dashed p-4"
    >
      Drop files here
    </div>
  );
}

