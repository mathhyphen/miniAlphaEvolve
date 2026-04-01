import { useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { EditorView } from '@codemirror/view'

// 代码编辑器属性类型
interface CodeEditorProps {
  code?: string
  readOnly?: boolean
}

// 代码编辑器组件 - 使用 react-codemirror，Python语法高亮
function CodeEditor({ code = '', readOnly = true }: CodeEditorProps) {
  // 默认代码
  const defaultCode = code || `# AlphaEvolve 生成的代码
def steiner_tree(graph, terminals):
    """
    计算Steiner最小树
    使用贪心算法近似求解
    """
    import math

    n = len(graph.nodes)
    if n == 0 or len(terminals) == 0:
        return []

    # 初始化
    dist = {node: math.inf for node in graph.nodes}
    parent = {node: None for node in graph.nodes}
    terminal_set = set(terminals)

    # Prim算法构建最小生成树
    current = terminals[0]
    dist[current] = 0
    visited = set()

    while len(visited) < n:
        visited.add(current)

        # 更新距离
        for neighbor in graph.neighbors(current):
            weight = graph[current][neighbor].get('weight', 1)
            if dist[neighbor] > weight and neighbor not in visited:
                dist[neighbor] = weight
                parent[neighbor] = current

        # 找最近未访问节点
        next_node = None
        min_dist = math.inf
        for node in graph.nodes:
            if node not in visited and dist[node] < min_dist:
                min_dist = dist[node]
                next_node = node

        if next_node is None:
            break
        current = next_node

    # 构建树边
    edges = []
    for node in graph.nodes:
        if parent[node] is not None:
            edges.append((parent[node], node))

    return edges
`

  const [displayCode] = useState<string>(defaultCode)

  // 自定义深色主题
  const customTheme = EditorView.theme({
    '&': {
      backgroundColor: 'rgba(0, 0, 0, 0.4)',
      color: '#fff',
      fontSize: '13px',
    },
    '.cm-content': {
      caretColor: '#fff',
      fontFamily: '"JetBrains Mono", "Fira Code", monospace',
    },
    '.cm-gutters': {
      backgroundColor: 'rgba(0, 0, 0, 0.2)',
      color: '#888',
      border: 'none',
    },
    '.cm-activeLineGutter': {
      backgroundColor: 'rgba(255, 255, 255, 0.1)',
    },
    '.cm-activeLine': {
      backgroundColor: 'rgba(255, 255, 255, 0.05)',
    },
    '.cm-selectionBackground': {
      backgroundColor: 'rgba(56, 239, 125, 0.3) !important',
    },
    '&.cm-focused .cm-selectionBackground': {
      backgroundColor: 'rgba(56, 239, 125, 0.3) !important',
    },
  })

  return (
    <div className="code-editor-wrapper">
      <CodeMirror
        value={displayCode}
        height="300px"
        theme={customTheme}
        extensions={[python()]}
        editable={!readOnly}
        basicSetup={{
          lineNumbers: true,
          highlightActiveLineGutter: true,
          highlightActiveLine: true,
          foldGutter: true,
        }}
      />
    </div>
  )
}

export default CodeEditor
