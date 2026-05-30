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

// 仪表盘主体组件 - 响应式布局，现代卡片设计
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
    <div className="space-y-6">
      {/* 统计卡片行 */}
      <StatsCards status={status} />

      {/* 主内容区 - 网格布局 */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* 演进曲线图表 - 全宽 */}
        <div className="xl:col-span-2 glass-card p-6 animate-fade-in">
          <EvolutionChart />
        </div>

        {/* 代码对比面板 */}
        <div className="glass-card p-6 animate-fade-in" style={{ animationDelay: '100ms' }}>
          <CodeCompare />
        </div>

        {/* 历史存档列表 */}
        <div className="glass-card p-6 animate-fade-in" style={{ animationDelay: '200ms' }}>
          <ArchiveList />
        </div>
      </div>
    </div>
  )
}

export default Dashboard
