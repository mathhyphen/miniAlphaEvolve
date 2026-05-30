import { useState, useEffect } from 'react'
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts'

// 图表数据类型
interface ChartData {
  generation: number
  bestScore: number
  avgScore: number
}

// 演进曲线图表组件
function EvolutionChart() {
  // 模拟数据
  const [data, setData] = useState<ChartData[]>([
    { generation: 0, bestScore: 0.45, avgScore: 0.32 },
    { generation: 1, bestScore: 0.48, avgScore: 0.35 },
    { generation: 2, bestScore: 0.52, avgScore: 0.38 },
    { generation: 3, bestScore: 0.55, avgScore: 0.40 },
    { generation: 4, bestScore: 0.58, avgScore: 0.43 },
    { generation: 5, bestScore: 0.62, avgScore: 0.45 },
    { generation: 6, bestScore: 0.65, avgScore: 0.48 },
    { generation: 7, bestScore: 0.68, avgScore: 0.50 },
  ])

  // 模拟实时更新
  useEffect(() => {
    const interval = setInterval(() => {
      setData((prev) => {
        const lastGen = prev[prev.length - 1]
        const newGen = lastGen.generation + 1
        const newBest = Math.min(1, lastGen.bestScore + (Math.random() * 0.03))
        const newAvg = Math.min(newBest - 0.1, lastGen.avgScore + (Math.random() * 0.02))

        // 保持最多20个数据点
        const newData = [...prev, { generation: newGen, bestScore: newBest, avgScore: newAvg }]
        return newData.slice(-20)
      })
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-card !bg-slate-900/90 !border-slate-600/30 px-4 py-3">
          <p className="text-slate-400 text-sm mb-2">Generation {label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center gap-2 text-sm">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-slate-300">{entry.name}:</span>
              <span className="font-semibold" style={{ color: entry.color }}>
                {(entry.value * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="animate-fade-in">
      {/* 标题区域 */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">Evolution Progress</h3>
          <p className="text-sm text-slate-400 mt-0.5">Real-time fitness tracking</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400" />
            <span className="text-sm text-slate-400">Best Score</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-indigo-400 to-purple-400" />
            <span className="text-sm text-slate-400">Average</span>
          </div>
        </div>
      </div>

      {/* 图表 */}
      <div className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
          >
            <defs>
              <linearGradient id="colorBest" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorAvg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(148, 163, 184, 0.1)"
              vertical={false}
            />
            <XAxis
              dataKey="generation"
              stroke="rgba(148, 163, 184, 0.5)"
              tick={{ fill: 'rgba(148, 163, 184, 0.7)', fontSize: 12 }}
              axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
              tickLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
              label={{
                value: 'Generation',
                position: 'bottom',
                offset: 5,
                fill: 'rgba(148, 163, 184, 0.5)',
                fontSize: 12,
              }}
            />
            <YAxis
              stroke="rgba(148, 163, 184, 0.5)"
              tick={{ fill: 'rgba(148, 163, 184, 0.7)', fontSize: 12 }}
              axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
              tickLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
              domain={[0, 1]}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="bestScore"
              name="Best Score"
              stroke="#34d399"
              strokeWidth={2.5}
              fill="url(#colorBest)"
              dot={{ fill: '#34d399', r: 4, strokeWidth: 0 }}
              activeDot={{ r: 6, fill: '#34d399', stroke: '#fff', strokeWidth: 2 }}
            />
            <Area
              type="monotone"
              dataKey="avgScore"
              name="Average"
              stroke="#818cf8"
              strokeWidth={2.5}
              fill="url(#colorAvg)"
              dot={{ fill: '#818cf8', r: 4, strokeWidth: 0 }}
              activeDot={{ r: 6, fill: '#818cf8', stroke: '#fff', strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default EvolutionChart
