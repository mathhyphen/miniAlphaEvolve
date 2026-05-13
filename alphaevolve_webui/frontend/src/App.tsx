import { useState, useCallback } from 'react'
import { Play, Square, Settings } from 'lucide-react'
import Dashboard from './components/Dashboard'
import { startEvolution, stopEvolution } from './api/client'

// 问题选项类型
interface Problem {
  id: string
  name: string
  description: string
}

// Benchmark选项类型
interface Benchmark {
  id: string
  name: string
}

// 应用主组件
function App() {
  // 状态管理
  const [isRunning, setIsRunning] = useState(false)
  const [selectedProblem, setSelectedProblem] = useState<string>('')
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('')
  const [problems] = useState<Problem[]>([
    { id: 'steiner', name: 'Steiner Tree Problem', description: 'Find minimum Steiner tree' },
    { id: 'partition', name: 'Graph Partition', description: 'Divide graph into balanced groups' },
  ])
  const [benchmarks] = useState<Benchmark[]>([
    { id: 'runtime', name: 'Runtime Performance' },
    { id: 'memory', name: 'Memory Usage' },
    { id: 'quality', name: 'Solution Quality' },
  ])

  // 开始演进
  const handleStart = useCallback(async () => {
    if (!selectedProblem || !selectedBenchmark) {
      alert('请选择问题和Benchmark')
      return
    }

    try {
      await startEvolution(selectedProblem, selectedBenchmark)
      setIsRunning(true)
    } catch (error) {
      console.error('启动演进失败:', error)
      alert('启动演进失败')
    }
  }, [selectedProblem, selectedBenchmark])

  // 停止演进
  const handleStop = useCallback(async () => {
    try {
      await stopEvolution()
      setIsRunning(false)
    } catch (error) {
      console.error('停止演进失败:', error)
    }
  }, [])

  return (
    <div className="app-container">
      {/* 顶部控制栏 */}
      <div className="glass-card control-bar fade-in">
        <h1>AlphaEvolve</h1>

        <div className="controls-left">
          {/* 问题选择器 */}
          <select
            value={selectedProblem}
            onChange={(e) => setSelectedProblem(e.target.value)}
            disabled={isRunning}
          >
            <option value="">选择问题</option>
            {problems.map((problem) => (
              <option key={problem.id} value={problem.id}>
                {problem.name}
              </option>
            ))}
          </select>

          {/* Benchmark选择器 */}
          <select
            value={selectedBenchmark}
            onChange={(e) => setSelectedBenchmark(e.target.value)}
            disabled={isRunning}
          >
            <option value="">选择Benchmark</option>
            {benchmarks.map((benchmark) => (
              <option key={benchmark.id} value={benchmark.id}>
                {benchmark.name}
              </option>
            ))}
          </select>
        </div>

        <div className="controls-right">
          {/* 开始/停止按钮 */}
          {!isRunning ? (
            <button className="btn btn-primary" onClick={handleStart}>
              <Play size={18} />
              开始演进
            </button>
          ) : (
            <button className="btn btn-danger" onClick={handleStop}>
              <Square size={18} />
              停止
            </button>
          )}

          {/* 设置按钮 */}
          <button className="btn btn-secondary">
            <Settings size={18} />
          </button>
        </div>
      </div>

      {/* 仪表盘主体 */}
      <Dashboard isRunning={isRunning} />
    </div>
  )
}

export default App
