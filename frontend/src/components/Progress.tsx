import React from 'react'

export default function Progress({ progress=0, message='' }:{progress:number, message?:string}){
  const pct = Math.round(progress*100)
  return (
    <div className="w-full">
      <div className="flex justify-between items-end mb-1">
        <span className="text-sm text-gray-600">{message || 'Processing...'}</span>
        <span className="text-xs text-gray-500">{pct}%</span>
      </div>
      <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
        <div className="h-full bg-black transition-all" style={{width: `${pct}%`}}/>
      </div>
    </div>
  )
}
