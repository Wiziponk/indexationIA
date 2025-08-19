import React from 'react'
import Chart from '../components/Chart'
import { api } from '../lib/api'

export default function Results({result}:{result:{result_path?:string, raw_path?:string, emb_path?:string, silhouette?:number|null, n_clusters?:number}}){
  const [rows, setRows] = React.useState<any[]>([])
  const path = result.result_path || ''
  React.useEffect(()=>{
    (async ()=>{
      if (!path) return
      // backend saved Parquet; we cannot read Parquet in browser — instead we just show a link.
    })()
  }, [path])

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-xl font-semibold mb-2">Downloads</h2>
        {result.raw_path && <a className="btn mr-2" href={`/download/${result.raw_path.split('/').pop()}`} target="_blank">Download RAW</a>}
        {result.emb_path && <a className="btn mr-2" href={`/download/${result.emb_path.split('/').pop()}`} target="_blank">Download EMB</a>}
        {result.result_path && <a className="btn" href={`/download/${result.result_path.split('/').pop()}`} target="_blank">Download RESULT</a>}
      </div>

      {typeof result.silhouette !== 'undefined' && (
        <div className="card">
          <div className="text-sm text-gray-600">Silhouette (if applicable): <b>{result.silhouette ?? 'n/a'}</b> — Clusters: <b>{result.n_clusters}</b></div>
        </div>
      )}

      <div className="text-sm text-gray-500">Tip: open the Parquet in Python or DuckDB to explore, or wire a small endpoint to sample rows for plotting.</div>
    </div>
  )
}
