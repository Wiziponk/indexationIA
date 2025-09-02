import { useState } from "react";
import StepFields from "./StepFields";
import StepScope from "./StepScope";
import StepTranscripts from "./StepTranscripts";
import Preview from "./Preview";
import Batch from "./Batch";

const STEPS = ["Fields", "Scope", "Transcripts", "Preview", "Batch"];

export default function Wizard() {
  const [step, setStep] = useState(0);

  const [primaryKey, setPrimaryKey] = useState("");
  const [embedFields, setEmbedFields] = useState<string[]>([]);

  const [mode, setMode] = useState<"api" | "excel">("api");
  const [excelToken, setExcelToken] = useState<string | null>(null);
  const [excelIdCol, setExcelIdCol] = useState<string | null>(null);

  const [transcripts, setTranscripts] = useState<File[]>([]);
  const [sampleIds, setSampleIds] = useState<string[]>([]);
  const [sampleId, setSampleId] = useState<string>("");
  const [prepared, setPrepared] = useState(false);
  const [previewed, setPreviewed] = useState(false);

  const [keepRatio, setKeepRatio] = useState(0.6);
  const [brief, setBrief] = useState("");
  const [withTitles, setWithTitles] = useState(true);

  const canNext =
    step === 0
      ? primaryKey !== "" && embedFields.length > 0
      : step === 1
      ? mode === "api" || (excelToken !== null && excelIdCol !== null)
      : step === 2
      ? prepared
      : step === 3
      ? previewed
      : false;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Dataset Wizard</h1>

      <div className="flex space-x-4">
        {STEPS.map((label, idx) => (
          <div
            key={label}
            className={`rounded px-3 py-1 text-sm ${
              idx === step ? "bg-primary text-primary-foreground" : "bg-muted"
            }`}
          >
            {idx + 1}. {label}
          </div>
        ))}
      </div>

      {step === 0 && (
        <StepFields
          primaryKey={primaryKey}
          setPrimaryKey={setPrimaryKey}
          embedFields={embedFields}
          setEmbedFields={setEmbedFields}
        />
      )}

      {step === 1 && (
        <StepScope
          mode={mode}
          setMode={setMode}
          token={excelToken}
          setToken={setExcelToken}
          idCol={excelIdCol}
          setIdCol={setExcelIdCol}
        />
      )}

      {step === 2 && (
        <StepTranscripts
          primaryKey={primaryKey}
          embedFields={embedFields}
          mode={mode}
          excelToken={excelToken}
          excelIdCol={excelIdCol}
          transcripts={transcripts}
          setTranscripts={setTranscripts}
          setSampleIds={setSampleIds}
          setSampleId={setSampleId}
          setPrepared={setPrepared}
        />
      )}

      {step === 3 && (
        <Preview
          primaryKey={primaryKey}
          embedFields={embedFields}
          mode={mode}
          excelToken={excelToken}
          excelIdCol={excelIdCol}
          transcripts={transcripts}
          sampleIds={sampleIds}
          sampleId={sampleId}
          setSampleId={setSampleId}
          keepRatio={keepRatio}
          setKeepRatio={setKeepRatio}
          brief={brief}
          setBrief={setBrief}
          withTitles={withTitles}
          setWithTitles={setWithTitles}
          setPreviewed={setPreviewed}
        />
      )}

      {step === 4 && (
        <Batch
          primaryKey={primaryKey}
          embedFields={embedFields}
          mode={mode}
          excelToken={excelToken}
          excelIdCol={excelIdCol}
          transcripts={transcripts}
          keepRatio={keepRatio}
          setKeepRatio={setKeepRatio}
          brief={brief}
          setBrief={setBrief}
          withTitles={withTitles}
          setWithTitles={setWithTitles}
        />
      )}

      <div className="flex justify-between pt-4">
        <button
          className="rounded border px-3 py-1"
          disabled={step === 0}
          onClick={() => setStep((s) => Math.max(0, s - 1))}
        >
          Back
        </button>
        {step < 4 && (
          <button
            className="rounded bg-primary px-3 py-1 text-primary-foreground disabled:opacity-50"
            disabled={!canNext}
            onClick={() => setStep((s) => s + 1)}
          >
            Next
          </button>
        )}
      </div>
    </div>
  );
}

