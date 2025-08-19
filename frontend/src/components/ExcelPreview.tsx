import React from 'react'

export default function ExcelPreview({head, columns}:{head:any[], columns:string[]}){
  return (
    <div className="overflow-auto max-h-64 border rounded-xl">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-100 sticky top-0">
          <tr>{columns.map(c => <th key={c} className="text-left p-2">{c}</th>)}</tr>
        </thead>
        <tbody>
          {head.map((row,i)=>(
            <tr key={i} className="odd:bg-white even:bg-gray-50">
              {columns.map(c => <td key={c} className="p-2">{String(row[c] ?? '')}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
