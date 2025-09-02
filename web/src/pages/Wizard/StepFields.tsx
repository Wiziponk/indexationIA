import { Dispatch, SetStateAction } from "react";
import { useApi } from "../../lib/useApi";

type FieldsResponse = { fields: string[] };

type Props = {
  primaryKey: string;
  setPrimaryKey: Dispatch<SetStateAction<string>>;
  embedFields: string[];
  setEmbedFields: Dispatch<SetStateAction<string[]>>;
};

export default function StepFields({
  primaryKey,
  setPrimaryKey,
  embedFields,
  setEmbedFields,
}: Props) {
  const { data, error, loading } = useApi<FieldsResponse>("/fields");

  const fields = data?.fields || [];

  return (
    <div className="space-y-4">
      <div>
        <h2 className="font-semibold">Primary key</h2>
        {loading && <p>Loading…</p>}
        {error && <p className="text-red-600">{error}</p>}
        {fields.length > 0 && (
          <select
            className="mt-1 w-full border p-1"
            value={primaryKey}
            onChange={(e) => setPrimaryKey(e.target.value)}
          >
            <option value="">Select…</option>
            {fields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        )}
      </div>

      <div>
        <h2 className="font-semibold">Embedding fields</h2>
        {fields.length > 0 && (
          <select
            multiple
            className="mt-1 w-full border p-1"
            value={embedFields}
            onChange={(e) =>
              setEmbedFields(
                Array.from(e.target.selectedOptions, (o) => o.value)
              )
            }
          >
            {fields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}

