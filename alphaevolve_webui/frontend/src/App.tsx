import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Activity,
  Braces,
  Database,
  Download,
  FlaskConical,
  GitBranch,
  Play,
  RefreshCcw,
  ShieldCheck,
  TerminalSquare,
} from 'lucide-react'
import {
  CandidateRecord,
  CustomCaseInput,
  RunSnapshot,
  TaskSpec,
  exportBestProgram,
  listTasks,
  startCustomRun,
  startRun,
} from './api/client'

type RunMode = 'builtin' | 'custom'

const DEFAULT_CUSTOM_PROGRAM = `def solve(value):
    return value
`

const DEFAULT_CUSTOM_CASES = `[
  {
    "case_id": "double_positive",
    "args": [3],
    "expected": 6,
    "description": "输入 3 时应返回 6"
  },
  {
    "case_id": "double_zero",
    "args": [0],
    "expected": 0,
    "description": "输入 0 时应返回 0"
  }
]`

function App() {
  const [tasks, setTasks] = useState<TaskSpec[]>([])
  const [selectedTaskId, setSelectedTaskId] = useState('')
  const [runMode, setRunMode] = useState<RunMode>('builtin')
  const [generations, setGenerations] = useState(4)
  const [archiveSize, setArchiveSize] = useState(6)
  const [run, setRun] = useState<RunSnapshot | null>(null)
  const [selectedCandidateId, setSelectedCandidateId] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState('')

  const [customTitle, setCustomTitle] = useState('把数字翻倍')
  const [customObjective, setCustomObjective] = useState('编写 solve(value)，返回输入数字的两倍。')
  const [customFunctionName, setCustomFunctionName] = useState('solve')
  const [customConstraints, setCustomConstraints] = useState('返回数字\n不要修改输入')
  const [customInitialProgram, setCustomInitialProgram] = useState(DEFAULT_CUSTOM_PROGRAM)
  const [customCasesJson, setCustomCasesJson] = useState(DEFAULT_CUSTOM_CASES)

  useEffect(() => {
    let mounted = true
    listTasks()
      .then((loadedTasks) => {
        if (!mounted) return
        setTasks(loadedTasks)
        setSelectedTaskId((current) => current || loadedTasks[0]?.task_id || '')
      })
      .catch((err: Error) => setError(err.message))
    return () => {
      mounted = false
    }
  }, [])

  const selectedTask = useMemo(
    () => tasks.find((task) => task.task_id === selectedTaskId) ?? null,
    [selectedTaskId, tasks],
  )

  const activeTask = run?.task ?? selectedTask

  const selectedCandidate = useMemo(() => {
    if (!run) return null
    return (
      run.history.find((candidate) => candidate.candidate_id === selectedCandidateId) ??
      run.best_candidate
    )
  }, [run, selectedCandidateId])

  const scoreSeries = useMemo(() => {
    if (!run) return []
    let best = 0
    return run.history.map((candidate) => {
      best = Math.max(best, candidate.score)
      return { generation: candidate.generation, score: candidate.score, best }
    })
  }, [run])

  async function handleStartRun() {
    setIsRunning(true)
    setError('')
    try {
      const nextRun =
        runMode === 'custom'
          ? await startCustomRun(buildCustomRunRequest())
          : await startRun({
              task_id: selectedTaskId,
              generations,
              archive_size: archiveSize,
            })
      setRun(nextRun)
      setSelectedCandidateId(nextRun.best_candidate?.candidate_id ?? '')
    } catch (err) {
      setError(err instanceof Error ? err.message : '运行失败')
    } finally {
      setIsRunning(false)
    }
  }

  async function handleExport() {
    if (!run) return
    try {
      const program = await exportBestProgram(run.run_id)
      const blob = new Blob([program], { type: 'text/x-python' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${run.run_id}-best.py`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : '导出失败')
    }
  }

  function buildCustomRunRequest() {
    const parsedCases = JSON.parse(customCasesJson) as CustomCaseInput[]
    if (!Array.isArray(parsedCases) || parsedCases.length === 0) {
      throw new Error('自定义问题至少需要一个评价用例')
    }

    return {
      title: customTitle,
      objective: customObjective,
      function_name: customFunctionName,
      initial_program: customInitialProgram,
      constraints: customConstraints
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean),
      cases: parsedCases,
      generations,
      archive_size: archiveSize,
    }
  }

  const canLaunch =
    runMode === 'custom'
      ? Boolean(customTitle && customObjective && customFunctionName && customInitialProgram && customCasesJson)
      : Boolean(selectedTaskId)

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">
            <Braces size={24} />
          </div>
          <div>
            <p className="eyebrow">算法发现工作台</p>
            <h1>AlphaEvolve</h1>
          </div>
        </div>
        <div className="run-actions">
          <button className="icon-button" onClick={() => void listTasks().then(setTasks)}>
            <RefreshCcw size={18} />
          </button>
          <button className="primary-button" disabled={isRunning || !canLaunch} onClick={handleStartRun}>
            <Play size={18} />
            {isRunning ? '运行中' : '启动搜索'}
          </button>
        </div>
      </header>

      {error && <div className="error-band">{error}</div>}

      <main className="workbench-grid">
        <section className="panel task-panel">
          <PanelTitle icon={<FlaskConical size={18} />} title="任务契约" />
          <div className="mode-tabs">
            <button className={runMode === 'builtin' ? 'active' : ''} onClick={() => setRunMode('builtin')}>
              内置任务
            </button>
            <button className={runMode === 'custom' ? 'active' : ''} onClick={() => setRunMode('custom')}>
              自定义问题
            </button>
          </div>

          {runMode === 'builtin' ? (
            <BuiltinTaskForm
              tasks={tasks}
              selectedTask={selectedTask}
              selectedTaskId={selectedTaskId}
              onSelectTask={setSelectedTaskId}
            />
          ) : (
            <CustomTaskForm
              title={customTitle}
              objective={customObjective}
              functionName={customFunctionName}
              constraints={customConstraints}
              initialProgram={customInitialProgram}
              casesJson={customCasesJson}
              onTitleChange={setCustomTitle}
              onObjectiveChange={setCustomObjective}
              onFunctionNameChange={setCustomFunctionName}
              onConstraintsChange={setCustomConstraints}
              onInitialProgramChange={setCustomInitialProgram}
              onCasesJsonChange={setCustomCasesJson}
            />
          )}

          <div className="control-grid">
            <NumberField label="进化代数" min={1} max={20} value={generations} onChange={setGenerations} />
            <NumberField label="归档容量" min={1} max={12} value={archiveSize} onChange={setArchiveSize} />
          </div>
        </section>

        <section className="panel run-panel">
          <PanelTitle icon={<Activity size={18} />} title="运行证据" />
          <div className="score-strip">
            <Metric label="状态" value={localizeStatus(run?.status ?? 'idle')} />
            <Metric label="代数" value={run ? `${run.current_generation}/${run.max_generations}` : '0/0'} />
            <Metric label="最佳分数" value={formatScore(run?.best_candidate?.score)} strong />
            <Metric label="归档" value={run?.archive.length.toString() ?? '0'} />
          </div>
          <ScoreChart points={scoreSeries} />
          <div className="candidate-table">
            <div className="table-head">
              <span>代数</span>
              <span>分数</span>
              <span>来源</span>
              <span>保留</span>
            </div>
            {(run?.history ?? []).map((candidate) => (
              <button
                key={candidate.candidate_id}
                className={`table-row ${selectedCandidate?.candidate_id === candidate.candidate_id ? 'active' : ''}`}
                onClick={() => setSelectedCandidateId(candidate.candidate_id)}
              >
                <span>第 {candidate.generation} 代</span>
                <span>{formatScore(candidate.score)}</span>
                <span>{localizeSource(candidate.proposal_source || 'seed')}</span>
                <span>{candidate.kept ? '是' : '否'}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel archive-panel">
          <PanelTitle icon={<Database size={18} />} title="程序归档" />
          <div className="archive-list">
            {(run?.archive ?? []).map((candidate) => (
              <ArchiveItem
                key={candidate.candidate_id}
                candidate={candidate}
                active={selectedCandidate?.candidate_id === candidate.candidate_id}
                onSelect={() => setSelectedCandidateId(candidate.candidate_id)}
              />
            ))}
          </div>
          <button className="secondary-button" disabled={!run?.best_candidate} onClick={handleExport}>
            <Download size={16} />
            导出最佳程序
          </button>
        </section>

        <section className="panel code-panel">
          <PanelTitle icon={<TerminalSquare size={18} />} title="最佳程序" />
          <pre className="code-block">
            <code>{selectedCandidate?.program ?? activeTask?.initial_program ?? customInitialProgram}</code>
          </pre>
        </section>

        <section className="panel prompt-panel">
          <PanelTitle icon={<GitBranch size={18} />} title="提示词与诊断" />
          <div className="diagnostic-grid">
            <pre className="prompt-block">{selectedCandidate?.prompt || '种子候选程序'}</pre>
            <div className="failure-list">
              {(selectedCandidate?.evaluation.failures ?? []).length === 0 ? (
                <div className="pass-box">所有评价用例均已通过</div>
              ) : (
                selectedCandidate?.evaluation.failures.map((failure) => (
                  <div key={failure.case_id} className="failure-card">
                    <strong>{failure.case_id}</strong>
                    <span>{failure.message}</span>
                    <small>期望值 {failure.expected}</small>
                    <small>实际值 {failure.actual}</small>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

function BuiltinTaskForm({
  tasks,
  selectedTask,
  selectedTaskId,
  onSelectTask,
}: {
  tasks: TaskSpec[]
  selectedTask: TaskSpec | null
  selectedTaskId: string
  onSelectTask: (taskId: string) => void
}) {
  return (
    <>
      <label className="field-label" htmlFor="task-select">
        任务
      </label>
      <select id="task-select" value={selectedTaskId} onChange={(event) => onSelectTask(event.target.value)}>
        {tasks.map((task) => (
          <option key={task.task_id} value={task.task_id}>
            {localizeTaskTitle(task)}
          </option>
        ))}
      </select>

      {selectedTask && (
        <>
          <div className="objective-block">
            <span>目标</span>
            <p>{localizeTaskObjective(selectedTask)}</p>
          </div>
          <div className="constraint-list">
            {selectedTask.constraints.map((constraint) => (
              <div key={constraint} className="constraint-row">
                <ShieldCheck size={14} />
                <span>{localizeConstraint(constraint)}</span>
              </div>
            ))}
          </div>
          <div className="metric-grid">
            <Metric label="评价用例" value={selectedTask.cases.length.toString()} />
            <Metric label="函数入口" value={selectedTask.function_name} />
            <Metric label="评分指标" value={localizeMetric(selectedTask.metric)} />
          </div>
        </>
      )}
    </>
  )
}

function CustomTaskForm({
  title,
  objective,
  functionName,
  constraints,
  initialProgram,
  casesJson,
  onTitleChange,
  onObjectiveChange,
  onFunctionNameChange,
  onConstraintsChange,
  onInitialProgramChange,
  onCasesJsonChange,
}: {
  title: string
  objective: string
  functionName: string
  constraints: string
  initialProgram: string
  casesJson: string
  onTitleChange: (value: string) => void
  onObjectiveChange: (value: string) => void
  onFunctionNameChange: (value: string) => void
  onConstraintsChange: (value: string) => void
  onInitialProgramChange: (value: string) => void
  onCasesJsonChange: (value: string) => void
}) {
  return (
    <div className="custom-form">
      <TextField label="问题名称" value={title} onChange={onTitleChange} />
      <TextArea label="问题目标" value={objective} rows={3} onChange={onObjectiveChange} />
      <TextField label="函数入口" value={functionName} onChange={onFunctionNameChange} />
      <TextArea label="约束条件（每行一条）" value={constraints} rows={3} onChange={onConstraintsChange} />
      <TextArea label="初始程序" value={initialProgram} rows={7} mono onChange={onInitialProgramChange} />
      <TextArea label="评价用例 JSON" value={casesJson} rows={10} mono onChange={onCasesJsonChange} />
      <p className="form-note">
        自定义问题必须提供可自动判断的评价用例。每个用例包含 case_id、args、expected，可选 validator: exact、approx 或 contains。
      </p>
    </div>
  )
}

function TextField({
  label,
  value,
  onChange,
}: {
  label: string
  value: string
  onChange: (value: string) => void
}) {
  return (
    <label className="text-field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  )
}

function TextArea({
  label,
  value,
  rows,
  mono = false,
  onChange,
}: {
  label: string
  value: string
  rows: number
  mono?: boolean
  onChange: (value: string) => void
}) {
  return (
    <label className="text-field">
      <span>{label}</span>
      <textarea
        className={mono ? 'mono-input' : ''}
        rows={rows}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  )
}

function PanelTitle({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="panel-title">
      {icon}
      <h2>{title}</h2>
    </div>
  )
}

function Metric({ label, value, strong = false }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong className={strong ? 'metric-strong' : ''}>{value}</strong>
    </div>
  )
}

function NumberField({
  label,
  min,
  max,
  value,
  onChange,
}: {
  label: string
  min: number
  max: number
  value: number
  onChange: (value: number) => void
}) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  )
}

function ArchiveItem({
  candidate,
  active,
  onSelect,
}: {
  candidate: CandidateRecord
  active: boolean
  onSelect: () => void
}) {
  return (
    <button className={`archive-item ${active ? 'active' : ''}`} onClick={onSelect}>
      <span className="archive-generation">第 {candidate.generation} 代</span>
      <span className="archive-score">{formatScore(candidate.score)}</span>
      <span className="archive-parent">{candidate.parent_id ? `父代 ${shortId(candidate.parent_id)}` : '种子程序'}</span>
    </button>
  )
}

function ScoreChart({ points }: { points: Array<{ generation: number; score: number; best: number }> }) {
  if (points.length === 0) {
    return <div className="empty-chart">尚未加载运行结果</div>
  }

  const width = 720
  const height = 180
  const maxGeneration = Math.max(...points.map((point) => point.generation), 1)
  const line = points
    .map((point) => {
      const x = (point.generation / maxGeneration) * (width - 28) + 14
      const y = height - point.best * (height - 28) - 14
      return `${x},${y}`
    })
    .join(' ')

  return (
    <svg className="score-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="各代最佳分数">
      <line x1="14" y1={height - 14} x2={width - 14} y2={height - 14} />
      <line x1="14" y1="14" x2="14" y2={height - 14} />
      <polyline points={line} />
      {points.map((point) => {
        const x = (point.generation / maxGeneration) * (width - 28) + 14
        const y = height - point.best * (height - 28) - 14
        return <circle key={`${point.generation}-${point.best}`} cx={x} cy={y} r="4" />
      })}
    </svg>
  )
}

function formatScore(score?: number | null) {
  if (score === undefined || score === null) return '0.000'
  return score.toFixed(3)
}

function shortId(value: string) {
  return value.slice(-6)
}

const TASK_TITLE_ZH: Record<string, string> = {
  sort_numbers: '数字排序',
  find_first_index: '查找首个索引',
  binary_search: '二分查找',
  two_sum_indices: '两数之和索引',
  valid_parentheses: '有效括号',
  fibonacci: '斐波那契数',
  gcd: '最大公约数',
  is_prime: '质数判断',
  sieve_primes: '埃拉托色尼筛法',
  factorial: '阶乘',
  reverse_string: '字符串反转',
  palindrome_check: '回文判断',
  merge_intervals: '区间合并',
  max_subarray_sum: '最大子数组和',
  longest_common_subsequence: '最长公共子序列',
  edit_distance: '编辑距离',
  knapsack_01: '0/1 背包',
  bfs_order: '广度优先搜索',
  dijkstra_shortest_path: 'Dijkstra 最短路径',
  matrix_multiply: '矩阵乘法优化',
}

const TASK_OBJECTIVE_ZH: Record<string, string> = {
  sort_numbers: '返回一个新的列表，将输入数字按升序排列。',
  find_first_index: '返回目标值在列表中第一次出现的位置；如果不存在，返回 -1。',
  binary_search: '在升序列表中返回目标值索引；如果不存在，返回 -1。',
  two_sum_indices: '返回两个不同元素的索引，使它们对应的数值之和等于目标值。',
  valid_parentheses: '判断括号字符串是否平衡且嵌套正确。',
  fibonacci: '返回第 n 个斐波那契数，F(0)=0，F(1)=1。',
  gcd: '返回两个整数的最大公约数。',
  is_prime: '判断整数 n 是否为质数。',
  sieve_primes: '返回小于等于 n 的全部质数。',
  factorial: '返回非负整数 n 的阶乘。',
  reverse_string: '返回输入字符串的反转结果。',
  palindrome_check: '判断字符串正读和反读是否完全相同。',
  merge_intervals: '合并重叠或相接的区间，并返回有序的不重叠区间。',
  max_subarray_sum: '返回非空连续子数组的最大和。',
  longest_common_subsequence: '返回两个字符串的最长公共子序列长度。',
  edit_distance: '返回两个字符串之间的 Levenshtein 编辑距离。',
  knapsack_01: '在容量限制内，每个物品最多使用一次，返回最大价值。',
  bfs_order: '从起点开始，按广度优先顺序返回访问到的节点。',
  dijkstra_shortest_path: '在非负权图中返回起点到可达节点的最短距离。',
  matrix_multiply: '实现 solve(a, b) 完成矩阵乘法，在保持整数结果完全正确的前提下优化纯 Python 运行速度。',
}

const CONSTRAINT_ZH: Record<string, string> = {
  'Do not mutate the input list.': '不要修改输入列表。',
  'Return a Python list.': '返回 Python 列表。',
  'Handle empty lists and duplicate values.': '正确处理空列表和重复值。',
  'Return an integer index.': '返回整数索引。',
  'Prefer the first matching index.': '优先返回第一个匹配位置。',
  'Do not raise when the target is missing.': '目标值不存在时不要抛出异常。',
  'Return a two-item list of indices.': '返回包含两个索引的列表。',
  'Do not reuse the same element twice.': '不能重复使用同一个元素。',
  'Return an empty list if no pair exists.': '不存在可行组合时返回空列表。',
  'Return a nested Python list.': '返回嵌套 Python 列表。',
  'Do not import numpy or external libraries.': '不要导入 numpy 或其他外部库。',
  'Support rectangular matrices with compatible dimensions.': '支持维度兼容的矩形矩阵。',
  'Correctness is mandatory before speed is rewarded.': '必须先保证正确性，之后才奖励速度。',
}

function localizeTaskTitle(task: TaskSpec) {
  return TASK_TITLE_ZH[task.task_id] ?? task.title
}

function localizeTaskObjective(task: TaskSpec) {
  return TASK_OBJECTIVE_ZH[task.task_id] ?? task.objective
}

function localizeConstraint(constraint: string) {
  return CONSTRAINT_ZH[constraint] ?? constraint
}

function localizeStatus(status: string) {
  const labels: Record<string, string> = {
    idle: '空闲',
    completed: '已完成',
    running: '运行中',
    failed: '失败',
  }
  return labels[status] ?? status
}

function localizeSource(source: string) {
  const labels: Record<string, string> = {
    seed: '种子',
    heuristic: '本地启发式',
    'heuristic-noop': '本地启发式',
    minimax: 'MiniMax 模型',
    'MiniMax-M2.7-highspeed': 'MiniMax-M2.7-highspeed',
  }
  return labels[source] ?? source
}

function localizeMetric(metric: string) {
  const labels: Record<string, string> = {
    pass_rate: '用例通过率',
    correctness_speed: '正确性 + 速度',
  }
  return labels[metric] ?? metric
}

export default App
