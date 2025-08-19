import React from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function Chart({rows}:{rows:{_x:number,_y:number,_cluster:number}[]}){
  return (
    <div className="h-[420px]">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart>
          <CartesianGrid />
          <XAxis type="number" dataKey="_x" />
          <YAxis type="number" dataKey="_y" />
          <Tooltip />
          <Scatter data={rows} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
