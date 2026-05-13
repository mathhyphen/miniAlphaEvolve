import { useState, useEffect } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
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

  return (
    <>
      <div className="chart-title">进化曲线</div>
      <ResponsiveContainer width="100%" height={250}>
      <LineChart
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
        <XAxis
          dataKey="generation"
          stroke="rgba(255,255,255,0.7)"
          tick={{ fill: 'rgba(255,255,255,0.7)' }}
        />
        <YAxis
          stroke="rgba(255,255,255,0.7)"
          tick={{ fill: 'rgba(255,255,255,0.7)' }}
          domain={[0, 1]}
          tickFormatter={(value) => (value * 100).toFixed(0) + '%'}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgba(0,0,0,0.8)',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
          }}
          labelStyle={{ color: '#fff' }}
          formatter={(value: number) => [(value * 100).toFixed(1) + '%']}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="bestScore"
          name="最佳分数"
          stroke="#38ef7d"
          strokeWidth={2}
          dot={{ fill: '#38ef7d', r: 4 }}
          activeDot={{ r: 6 }}
        />
        <Line
          type="monotone"
          dataKey="avgScore"
          name="平均分数"
          stroke="#00d9ff"
          strokeWidth={2}
          dot={{ fill: '#00d9ff', r: 4 }}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
    </>
  )
}

export default EvolutionChart
