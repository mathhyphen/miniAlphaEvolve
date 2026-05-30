const API_BASE = '/api'

export interface EvaluationCase {
  case_id: string
  args: unknown[]
  expected: unknown
  description: string
  validator: string
}

export interface TaskSpec {
  task_id: string
  title: string
  objective: string
  initial_program: string
  function_name: string
  language: string
  metric: string
  constraints: string[]
  tags: string[]
  cases: EvaluationCase[]
  baseline_program: string | null
  benchmark_repetitions: number
}

export interface CaseFailure {
  case_id: string
  message: string
  expected: string
  actual: string
}

export interface EvaluationReport {
  score: number
  passed_cases: number
  total_cases: number
  failures: CaseFailure[]
  execution_time: number
  peak_memory_mb: number
  metrics: Record<string, unknown>
}

export interface CandidateRecord {
  candidate_id: string
  generation: number
  parent_id: string | null
  program: string
  score: number
  evaluation: EvaluationReport
  prompt: string
  proposal: string
  proposal_source: string
  kept: boolean
  created_at: string
}

export interface RunSnapshot {
  run_id: string
  task: TaskSpec
  status: string
  current_generation: number
  max_generations: number
  best_candidate: CandidateRecord | null
  archive: CandidateRecord[]
  history: CandidateRecord[]
  created_at: string
  updated_at: string
}

interface ApiEnvelope<T> {
  success: boolean
  data: T
  error: string | null
}

export interface RunRequest {
  task_id: string
  generations: number
  archive_size: number
}

export interface CustomCaseInput {
  case_id: string
  args: unknown[]
  expected: unknown
  description?: string
  validator?: 'exact' | 'approx' | 'contains'
}

export interface CustomRunRequest {
  title: string
  objective: string
  initial_program: string
  function_name: string
  constraints: string[]
  cases: CustomCaseInput[]
  generations: number
  archive_size: number
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
    ...options,
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || `API request failed with ${response.status}`)
  }

  const payload = (await response.json()) as ApiEnvelope<T>
  if (!payload.success) {
    throw new Error(payload.error ?? 'API request failed')
  }
  return payload.data
}

export function listTasks(): Promise<TaskSpec[]> {
  return request<TaskSpec[]>('/tasks')
}

export function startRun(input: RunRequest): Promise<RunSnapshot> {
  return request<RunSnapshot>('/runs', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function startCustomRun(input: CustomRunRequest): Promise<RunSnapshot> {
  return request<RunSnapshot>('/runs/custom', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function getRun(runId: string): Promise<RunSnapshot> {
  return request<RunSnapshot>(`/runs/${runId}`)
}

export async function exportBestProgram(runId: string): Promise<string> {
  const response = await fetch(`${API_BASE}/runs/${runId}/export`)
  if (!response.ok) {
    throw new Error(`Export failed with ${response.status}`)
  }
  return response.text()
}
