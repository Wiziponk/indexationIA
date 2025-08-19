import axios from 'axios'

const base = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
export const api = axios.create({ baseURL: base })

export async function getFields(baseUrl?: string) {
  const { data } = await api.get('/api/fields', { params: { base: baseUrl } })
  return data
}

export async function getSample(baseUrl: string | undefined, fields: string[]) {
  const { data } = await api.get('/api/sample', { params: { base: baseUrl, fields: fields.join(',') } })
  return data
}

export async function uploadExcel(file: File) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/preview-excel', form, { headers: { 'Content-Type': 'multipart/form-data' } })
  return data
}

export async function startGenerate(payload: any) {
  const { data } = await api.post('/_jobs/generate', payload)
  return data.job_id as string
}
export async function startCluster(payload: any) {
  const { data } = await api.post('/_jobs/cluster', payload)
  return data.job_id as string
}
export async function jobStatus(jobId: string) {
  const { data } = await api.get('/_jobs/' + jobId)
  return data
}
export async function listArtifacts() {
  const { data } = await api.get('/artifacts')
  return data.files as string[]
}
