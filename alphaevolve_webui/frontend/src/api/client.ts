// API 客户端 - 与后端通信
// 基础URL - 开发环境使用Vite代理，生产环境应配置为实际后端地址
const API_BASE = '/api'

// 请求选项类型
interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
  body?: unknown
  headers?: Record<string, string>
}

// 通用请求函数
async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, headers = {} } = options

  const config: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  }

  if (body) {
    config.body = JSON.stringify(body)
  }

  const response = await fetch(`${API_BASE}${endpoint}`, config)

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`)
  }

  return response.json()
}

// 问题类型
export interface Problem {
  id: string
  name: string
  description: string
}

// Benchmark类型
export interface Benchmark {
  id: string
  name: string
}

// 演进状态类型
export interface EvolutionStatus {
  isRunning: boolean
  generation: number
  bestScore: number
  avgScore: number
  candidates: number
  diversity: number
}

// 存档项类型
export interface ArchiveItem {
  id: string
  generation: number
  score: number
  code: string
  timestamp: string
  parentId?: string
}

// 历史数据点类型
export interface DataPoint {
  generation: number
  bestScore: number
  avgScore: number
}

// 获取问题列表
export async function getProblems(): Promise<Problem[]> {
  return request<Problem[]>('/problems')
}

// 获取Benchmark列表
export async function getBenchmarks(): Promise<Benchmark[]> {
  return request<Benchmark[]>('/benchmarks')
}

// 获取演进状态
export async function getStatus(): Promise<EvolutionStatus> {
  return request<EvolutionStatus>('/status')
}

// 开始演进
export async function startEvolution(problemId: string, benchmarkId: string): Promise<void> {
  await request('/evolution/start', {
    method: 'POST',
    body: { problemId, benchmarkId },
  })
}

// 停止演进
export async function stopEvolution(): Promise<void> {
  await request('/evolution/stop', { method: 'POST' })
}

// 获取历史存档列表
export async function getArchive(): Promise<ArchiveItem[]> {
  return request<ArchiveItem[]>('/archive')
}

// 获取指定存档的代码
export async function getArchiveCode(id: string): Promise<{ code: string; version: number }> {
  return request(`/archive/${id}`)
}

// 获取演进曲线数据
export async function getEvolutionData(): Promise<DataPoint[]> {
  return request<DataPoint[]>('/evolution/data')
}

// 导出存档
export async function exportArchive(id: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/archive/${id}/export`)
  if (!response.ok) {
    throw new Error('导出失败')
  }
  return response.blob()
}

// API客户端对象 - 方便统一调用
export const api = {
  getProblems,
  getBenchmarks,
  getStatus,
  startEvolution,
  stopEvolution,
  getArchive,
  getArchiveCode,
  getEvolutionData,
  exportArchive,
}
