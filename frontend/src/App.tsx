import React from 'react'
import Generate from './pages/Generate'
import Cluster from './pages/Cluster'
import Results from './pages/Results'

export default function App(){
  const [genResult, setGenResult] = React.useState<any|null>(null)
  const [clusResult, setClusResult] = React.useState<any|null>(null)
  return (
    <main className="max-w-5xl mx-auto p-6 space-y-6">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Embeddings & Clustering</h1>
      </header>

      {!genResult && <Generate onDone={setGenResult}/>}
      {genResult && !clusResult && <Cluster onDone={setClusResult}/>}
      {clusResult && <Results result={{...genResult, ...clusResult}}/>}
    </main>
  )
}
