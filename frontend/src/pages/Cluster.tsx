import React from 'react'
import { listArtifacts, startCluster, jobStatus } from '../lib/api'
import Progress from '../components/Progress'

export default function Cluster({onDone}:{onDone:(res:{result_path:string, silhouette:number|null, n_clusters:number})=>void}){
  const [files, setFiles] = React.useState<string[]>([])
  const [raw, setRaw] = React.useState<string>('')
  const [emb, setEmb] = React.useState<string>('')
  const [algo, setAlgo] = React.useState<'kmeans'|'dbscan'>('kmeans')
  const [kmin, setKmin] = React.useState(6)
  const [kmax, setKmax] = React.useState(12)
  const [eps, setEps] = React.useState(0.5)
  const [mins, setMins] = React.useState(5)
  const [loading, setLoading] = React.useState(false)
  const [progress, setProgress] = React.useState(0)
  const [message, setMessage] = React.useState('')

  React.useEffect(()=>{ (async ()=>{
    setFiles(await listArtifacts())
  })() }, [])

  async function run(){
    setLoading(true)
    const job = await startCluster({
      raw_path: `/data/${raw}`.replace('/data/','/data/'), // path is inside container; backend reads its volume
      emb_path: `/data/${emb}`.replace('/data/','/data/'),
      algorithm: algo,
      k_min: kmin,
      k_max: kmax,
      eps, min_samples: mins
    })
    const t = setInterval(async ()=>{
      const s = await jobStatus(job)
      setProgress(s.progress || 0)
      setMessage(s.message || 'Running')
      if (s.status === 'finished'){
        clearInterval(t)
        setLoading(false)
        onDone(s.result)
      } else if (s.status === 'failed'){
        clearInterval(t)
        setLoading(false)
        alert('Job failed')
      }
    }, 1000)
  }

  return (
    <div className="space-y-6">
      <div className="card space-y-3">
        <h2 className="text-xl font-semibold">Select dataset & embeddings</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-sm text-gray-600 mb-1">raw_*.parquet</label>
            <select className="border rounded-xl px-3 py-2 w-full" value={raw} onChange={e=>setRaw(e.target.value)}>
              <option value="">-- choose --</option>
              {files.filter(f=>f.startsWith('raw_')).map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-600 mb-1">emb_*.npy</label>
            <select className="border rounded-xl px-3 py-2 w-full" value={emb} onChange={e=>setEmb(e.target.value)}>
              <option value="">-- choose --</option>
              {files.filter(f=>f.startswith('emb_')).map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div className="card space-y-3">
        <h2 className="text-xl font-semibold">Algorithm</h2>
        <div className="flex gap-3 items-center">
          <label className="flex items-center gap-2">
            <input type="radio" checked={algo==='kmeans'} onChange={()=>setAlgo('kmeans')} /> K-Means
          </label>
          <label className="flex items-center gap-2">
            <input type="radio" checked={algo==='dbscan'} onChange={()=>setAlgo('dbscan')} /> DBSCAN
          </label>
        </div>

        {algo==='kmeans' ? (
          <div className="grid grid-cols-2 gap-3">
            <div><label className="text-sm">k_min</label><input className="border rounded-xl px-3 py-2 w-full" type="number" value={kmin} onChange={e=>setKmin(parseInt(e.target.value))}/></div>
            <div><label className="text-sm">k_max</label><input className="border rounded-xl px-3 py-2 w-full" type="number" value={kmax} onChange={e=>setKmax(parseInt(e.target.value))}/></div>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            <div><label className="text-sm">eps</label><input className="border rounded-xl px-3 py-2 w-full" type="number" step="0.1" value={eps} onChange={e=>setEps(parseFloat(e.target.value))}/></div>
            <div><label className="text-sm">min_samples</label><input className="border rounded-xl px-3 py-2 w-full" type="number" value={mins} onChange={e=>setMins(parseInt(e.target.value))}/></div>
          </div>
        )}
      </div>

      <div className="card space-y-3">
        <h2 className="text-xl font-semibold">Run</h2>
        {!loading ? (
          <button className="btn" disabled={!raw || !emb} onClick={run}>Cluster</button>
        ) : (
          <Progress progress={progress} message={message}/>
        )}
      </div>
    </div>
  )
}
