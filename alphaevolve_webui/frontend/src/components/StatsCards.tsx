import { Trophy, Users, Sparkles, GitBranch } from 'lucide-react'

// 状态类型
interface EvolutionStatus {
  isRunning: boolean
  generation: number
  bestScore: number
  avgScore: number
  candidates: number
  diversity: number
}

interface StatsCardsProps {
  status: EvolutionStatus
}

// 统计卡片组件
function StatsCards({ status }: StatsCardsProps) {
  const cards = [
    {
      icon: <Trophy size={24} />,
      value: (status.bestScore * 100).toFixed(1) + '%',
      label: '最佳分数',
      color: '#ffd700',
    },
    {
      icon: <GitBranch size={24} />,
      value: status.generation.toString(),
      label: '进化代数',
      color: '#38ef7d',
    },
    {
      icon: <Users size={24} />,
      value: status.candidates.toString(),
      label: '候选数量',
      color: '#00d9ff',
    },
    {
      icon: <Sparkles size={24} />,
      value: (status.diversity * 100).toFixed(1) + '%',
      label: '多样性',
      color: '#f45c43',
    },
  ]

  return (
    <div className="stats-grid">
      {cards.map((card, index) => (
        <div key={index} className="stat-card glass-card">
          <div style={{ color: card.color, marginBottom: '8px' }}>
            {card.icon}
          </div>
          <div className="stat-value" style={{ color: card.color }}>
            {card.value}
          </div>
          <div className="stat-label">{card.label}</div>
        </div>
      ))}
    </div>
  )
}

export default StatsCards
