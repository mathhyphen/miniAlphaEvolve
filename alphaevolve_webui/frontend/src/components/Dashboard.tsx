import { useState, useEffect } from 'react'
import StatsCards from './StatsCards'
import EvolutionChart from './EvolutionChart'
import CodeCompare from './CodeCompare'
import ArchiveList from './ArchiveList'

// 状态类型
interface EvolutionStatus {
  isRunning: boolean
  generation: number
  bestScore: number
  avgScore: number
  candidates: number
  diversity: number
}

interface DashboardProps {
  isRunning: boolean
}

// 仪表盘主体组件 - 响应式布局，玻璃态卡片样式
function Dashboard({ isRunning }: DashboardProps) {
  // 本地状态管理 - 模拟数据
  const [status, setStatus] = useState<EvolutionStatus>({
    isRunning: false,
    generation: 0,
    bestScore: 0,
    avgScore: 0,
    candidates: 0,
    diversity: 0,
  })

  // 模拟实时数据更新
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      setStatus((prev) => ({
        ...prev,
        generation: prev.generation + 1,
        bestScore: Math.min(1, prev.bestScore + 0.02),
        avgScore: Math.min(0.85, prev.avgScore + 0.01),
        candidates: 50 + Math.floor(Math.random() * 20),
        diversity: 0.6 + Math.random() * 0.2,
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [isRunning])

  // 重置状态当演进停止时
  useEffect(() => {
    if (!isRunning) {
      setStatus({
        isRunning: false,
        generation: 0,
        bestScore: 0,
        avgScore: 0,
        candidates: 0,
        diversity: 0,
      })
    }
  }, [isRunning])

  return (
    <div className="dashboard">
      {/* 统计卡片行 */}
      <div className="dashboard-top">
        <StatsCards status={status} />
      </div>

      {/* 演进曲线图表 */}
      <div className="chart-container glass-card">
        <EvolutionChart />
      </div>

      {/* 代码对比面板 */}
      <div className="glass-card code-panel">
        <CodeCompare />
      </div>

      {/* 历史存档列表 */}
      <div className="glass-card">
        <ArchiveList />
      </div>
    </div>
  )
}

export default Dashboard
