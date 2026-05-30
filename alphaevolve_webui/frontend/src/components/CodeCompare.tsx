import { useState } from 'react'
import { ArrowRight, Lightbulb, Code2 } from 'lucide-react'

// 代码对比组件 - Side-by-side显示两个版本代码，现代设计
function CodeCompare() {
  // 当前版本代码
  const [currentCode] = useState<string>(`# AlphaEvolve v12 - Current Best
def steiner_tree_optimized(graph, terminals):
    """
    Optimized Steiner Tree Algorithm - Version 12
    Improvement: Dijkstra替代Prim，边压缩优化
    """
    import heapq
    from collections import defaultdict

    n = len(graph.nodes)
    if n == 0:
        return []

    # Dijkstra算法
    dist = {node: float('inf') for node in graph.nodes}
    parent = {node: None for node in graph.nodes}
    dist[terminals[0]] = 0

    pq = [(0, terminals[0])]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v in graph.neighbors(u):
            weight = graph[u][v].get('weight', 1)
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))

    # 构建最小生成树
    edges = [(parent[v], v) for v in graph.nodes if parent[v] is not None]
    return edges`)

  // 父版本代码
  const [parentCode] = useState<string>(`# AlphaEvolve v11 - Parent Version
def steiner_tree(graph, terminals):
    """
    Steiner Tree Algorithm - Version 11
    Using Prim's algorithm
    """
    import math

    n = len(graph.nodes)
    if n == 0:
        return []

    dist = {node: math.inf for node in graph.nodes}
    parent = {node: None for node in graph.nodes}
    dist[terminals[0]] = 0

    current = terminals[0]
    visited = set()

    while len(visited) < n:
        visited.add(current)

        for neighbor in graph.neighbors(current):
            weight = graph[current][neighbor].get('weight', 1)
            if dist[neighbor] > weight:
                dist[neighbor] = weight
                parent[neighbor] = current

        # 找最小距离节点
        next_node = min((n for n in graph.nodes if n not in visited),
                       default=None, key=lambda x: dist[x])
        if next_node is None:
            break
        current = next_node

    edges = [(parent[v], v) for v in graph.nodes if parent[v] is not None]
    return edges`)

  const improvements = [
    'Dijkstra algorithm instead of Prim (more stable convergence)',
    'Heap optimization, O((V+E)log V) complexity',
    'Early termination condition added',
  ]

  return (
    <div className="h-full flex flex-col">
      {/* 标题 */}
      <div className="flex items-center gap-3 mb-5">
        <div className="p-2 rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/30">
          <Code2 className="w-5 h-5 text-indigo-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-slate-100">Code Comparison</h3>
          <p className="text-xs text-slate-400">v11 → v12</p>
        </div>
      </div>

      {/* 代码对比网格 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1">
        {/* 父版本 */}
        <div className="relative group">
          <div className="absolute -top-3 left-4 z-10">
            <span className="badge badge-warning">v11 - Parent</span>
          </div>
          <div className="code-block p-4 h-full">
            <pre className="text-slate-400 text-xs leading-relaxed overflow-auto max-h-[220px]">
              <code>{parentCode}</code>
            </pre>
          </div>
        </div>

        {/* 当前版本 */}
        <div className="relative group">
          <div className="absolute -top-3 left-4 z-10">
            <span className="badge badge-success">v12 - Current</span>
          </div>
          <div className="code-block p-4 h-full border border-emerald-500/20">
            <pre className="text-emerald-400/90 text-xs leading-relaxed overflow-auto max-h-[220px]">
              <code>{currentCode}</code>
            </pre>
          </div>
        </div>
      </div>

      {/* 改进说明 */}
      <div className="mt-4 p-4 rounded-lg bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border border-emerald-500/20">
        <div className="flex items-center gap-2 mb-3">
          <Lightbulb className="w-4 h-4 text-amber-400" />
          <span className="text-sm font-semibold text-amber-400">Key Improvements</span>
        </div>
        <ul className="space-y-2">
          {improvements.map((item, index) => (
            <li key={index} className="flex items-start gap-2 text-sm">
              <ArrowRight className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
              <span className="text-slate-300">{item}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default CodeCompare
