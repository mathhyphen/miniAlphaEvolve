import { useEffect, useState } from 'react'
import { Trophy, GitBranch, Users, Sparkles } from 'lucide-react'

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

// 动画数字组件
function AnimatedNumber({ value, suffix = '', decimals = 0 }: { value: number; suffix?: string; decimals?: number }) {
  const [displayValue, setDisplayValue] = useState(0)

  useEffect(() => {
    const duration = 500
    const startValue = displayValue
    const endValue = value
    const startTime = Date.now()

    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      const easeOut = 1 - Math.pow(1 - progress, 3)
      const current = startValue + (endValue - startValue) * easeOut

      setDisplayValue(current)

      if (progress < 1) {
        requestAnimationFrame(animate)
      }
    }

    requestAnimationFrame(animate)
  }, [value])

  return (
    <span>
      {decimals > 0 ? displayValue.toFixed(decimals) : Math.floor(displayValue)}
      {suffix}
    </span>
  )
}

// 统计卡片组件
function StatsCards({ status }: StatsCardsProps) {
  const cards = [
    {
      icon: <Trophy className="w-6 h-6" />,
      value: status.bestScore * 100,
      suffix: '%',
      decimals: 1,
      label: 'Best Score',
      color: 'from-amber-400 to-orange-500',
      glowColor: 'shadow-amber-500/20',
      bgGradient: 'from-amber-500/10 to-orange-500/10',
      borderColor: 'border-amber-500/20',
    },
    {
      icon: <GitBranch className="w-6 h-6" />,
      value: status.generation,
      label: 'Generation',
      color: 'from-emerald-400 to-cyan-500',
      glowColor: 'shadow-emerald-500/20',
      bgGradient: 'from-emerald-500/10 to-cyan-500/10',
      borderColor: 'border-emerald-500/20',
    },
    {
      icon: <Users className="w-6 h-6" />,
      value: status.candidates,
      label: 'Candidates',
      color: 'from-cyan-400 to-blue-500',
      glowColor: 'shadow-cyan-500/20',
      bgGradient: 'from-cyan-500/10 to-blue-500/10',
      borderColor: 'border-cyan-500/20',
    },
    {
      icon: <Sparkles className="w-6 h-6" />,
      value: status.diversity * 100,
      suffix: '%',
      decimals: 1,
      label: 'Diversity',
      color: 'from-purple-400 to-pink-500',
      glowColor: 'shadow-purple-500/20',
      bgGradient: 'from-purple-500/10 to-pink-500/10',
      borderColor: 'border-purple-500/20',
    },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 stagger-children">
      {cards.map((card, index) => (
        <div
          key={index}
          className="glass-card glass-card-hover p-5 group relative overflow-hidden"
        >
          {/* 背景渐变 */}
          <div className={`absolute inset-0 bg-gradient-to-br ${card.bgGradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />

          {/* 边框光晕 */}
          <div className={`absolute inset-0 border ${card.borderColor} rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />

          <div className="relative z-10">
            <div className="flex items-center gap-3 mb-4">
              <div className={`p-2.5 rounded-xl bg-gradient-to-br ${card.color} shadow-lg ${card.glowColor}`}>
                <div className="text-white">
                  {card.icon}
                </div>
              </div>
              <span className="text-sm font-medium text-slate-400">{card.label}</span>
            </div>

            <div className={`text-4xl font-bold bg-gradient-to-r ${card.color} bg-clip-text text-transparent`}>
              <AnimatedNumber
                value={card.value}
                suffix={card.suffix || ''}
                decimals={card.decimals || 0}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default StatsCards
