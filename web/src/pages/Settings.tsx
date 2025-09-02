import { API_BASE } from "../lib/api";
import { useApi } from "../lib/useApi";

type HealthResponse = { ok: boolean; version: string };
type FieldsResponse = {
  fields: string[];
  api_base: string;
  source: string;
  sample_size: number;
  note?: string;
};

export default function Settings() {
  const {
    data: health,
    error: healthError,
    loading: healthLoading,
  } = useApi<HealthResponse>("/health");
  const {
    data: fields,
    error: fieldsError,
    loading: fieldsLoading,
  } = useApi<FieldsResponse>("/fields");

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">Settings</h1>

      <section>
        <h2 className="font-semibold">API</h2>
        <p>
          Base URL: <code>{API_BASE}</code>
        </p>
        {healthLoading && <p>Checking…</p>}
        {healthError && <p className="text-red-600">{healthError}</p>}
        {health && <p className="text-green-700">OK • {health.version}</p>}
      </section>

      <section>
        <h2 className="font-semibold">Fields</h2>
        {fieldsLoading && <p>Loading…</p>}
        {fieldsError && <p className="text-red-600">{fieldsError}</p>}
        {fields && (
          <div className="space-y-2">
            <p>
              {fields.fields.length} fields loaded from {fields.source} (sample
              {" "}
              {fields.sample_size})
            </p>
            <p>
              EDUC API base: <code>{fields.api_base || "n/a"}</code>
            </p>
            {fields.note && (
              <div className="mt-2 rounded border border-red-200 bg-red-100 p-2 text-sm text-red-800">
                {fields.note}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

