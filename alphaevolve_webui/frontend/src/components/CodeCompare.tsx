import { useState } from 'react'

// 代码对比组件 - Side-by-side显示两个版本代码，高亮差异行
function CodeCompare() {
  // 当前版本代码
  const [currentCode] = useState<string>(`# AlphaEvolve v12 - 当前最佳
def steiner_tree_optimized(graph, terminals):
    """
    优化的Steiner树算法 - 版本12
    改进: 使用Dijkstra替代Prim，边压缩优化
    """
    import heapq
    from collections import defaultdict

    n = len(graph.nodes)
    if n == 0:
        return []

    # 使用Dijkstra算法
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
    return edges
`)

  // 父版本代码
  const [parentCode] = useState<string>(`# AlphaEvolve v11 - 父版本
def steiner_tree(graph, terminals):
    """
    Steiner树算法 - 版本11
    使用Prim算法构建
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
    return edges
`)

  return (
    <div className="code-panel">
      <div className="code-title">代码对比</div>
      <div className="compare-grid">
        {/* 父版本 */}
        <div className="compare-section">
          <div className="compare-title">v11 - 父版本</div>
          <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '300px', color: '#a1a1aa' }}>
            <code>{parentCode}</code>
          </pre>
        </div>

        {/* 当前版本 */}
        <div className="compare-section">
          <div className="compare-title">v12 - 当前版本</div>
          <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '300px', color: '#38ef7d' }}>
            <code>{currentCode}</code>
          </pre>
        </div>
      </div>

      {/* 改进说明 */}
      <div style={{
        marginTop: '16px',
        padding: '12px',
        background: 'rgba(56, 239, 125, 0.1)',
        borderRadius: '8px',
        border: '1px solid rgba(56, 239, 125, 0.2)'
      }}>
        <div style={{ fontSize: '12px', color: '#38ef7d', fontWeight: 600, marginBottom: '4px' }}>
          改进点:
        </div>
        <ul style={{ fontSize: '12px', color: '#a1a1aa', marginLeft: '16px', lineHeight: 1.8 }}>
          <li>使用Dijkstra算法替代Prim (更稳定的收敛性)</li>
          <li>引入堆优化，时间复杂度 O((V+E)logV)</li>
          <li>添加早期终止条件</li>
        </ul>
      </div>
    </div>
  )
}

export default CodeCompare
