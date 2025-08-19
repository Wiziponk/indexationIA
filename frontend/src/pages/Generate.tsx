import React from 'react'
import { getFields, getSample, uploadExcel, startGenerate, jobStatus } from '../lib/api'
import Progress from '../components/Progress'

export default function Generate({onDone}:{onDone:(res:{raw_path:string, emb_path:string, count:number})=>void}){
  const [apiBase, setApiBase] = React.useState<string>('')
  const [fields, setFields] = React.useState<string[]>([])
  const [primary, setPrimary] = React.useState<string>('id')
  const [chosen, setChosen] = React.useState<string[]>([])
  const [excel, setExcel] = React.useState<{token:string, columns:string[], head:any[]} | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [progress, setProgress] = React.useState(0)
  const [message, setMessage] = React.useState('')

  async function loadFields(){
    const data = await getFields(apiBase || undefined)
    setFields(data.fields)
    if (!primary && data.fields.length) setPrimary(data.fields[0])
  }

  async function onExcel(e: React.ChangeEvent<HTMLInputElement>){
    const file = e.target.files?.[0]
    if (!file) return
    const data = await uploadExcel(file)
    setExcel(data)
  }

  async function run(){
    setLoading(true)
    const job = await startGenerate({
      base: apiBase || undefined,
      primary_key: primary,
      fields: chosen,
      excel_token: excel?.token || null
    })
    // poll
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
        <h2 className="text-xl font-semibold">1) Data source</h2>
        <div className="flex gap-2">
          <input className="border rounded-xl px-3 py-2 w-full" placeholder="API base (optional, uses sample if empty)" value={apiBase} onChange={e=>setApiBase(e.target.value)} />
          <button className="btn" onClick={loadFields}>Load fields</button>
        </div>
        {!!fields.length && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm text-gray-600 mb-1">Primary key</label>
              <select className="border rounded-xl px-3 py-2 w-full" value={primary} onChange={e=>setPrimary(e.target.value)}>
                {fields.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Fields to embed</label>
              <div className="flex flex-wrap gap-2">
                {fields.map(f => (
                  <label key={f} className={"px-3 py-2 rounded-xl border cursor-pointer " + (chosen.includes(f) ? "bg-black text-white" : "bg-white")}>
                    <input type="checkbox" className="hidden" checked={chosen.includes(f)} onChange={()=>{
                      setChosen(x => x.includes(f) ? x.filter(y=>y!==f) : [...x, f])
                    }}/>
                    {f}
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="card space-y-3">
        <h2 className="text-xl font-semibold">2) (Optional) Excel/CSV filter</h2>
        <input type="file" onChange={onExcel} />
        {excel && (
          <div>
            <p className="text-sm text-gray-600 mb-2">Preview (first 10 rows):</p>
            <div className="text-xs text-gray-500">Token: {excel.token}</div>
          </div>
        )}
      </div>

      <div className="card space-y-3">
        <h2 className="text-xl font-semibold">3) Generate embeddings</h2>
        {!loading ? (
          <button className="btn" disabled={!primary || !chosen.length} onClick={run}>Run</button>
        ) : (
          <Progress progress={progress} message={message}/>
        )}
      </div>
    </div>
  )
}
