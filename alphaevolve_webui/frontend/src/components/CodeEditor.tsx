import { useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { EditorView } from '@codemirror/view'

// 代码编辑器属性类型
interface CodeEditorProps {
  code?: string
  readOnly?: boolean
}

// 代码编辑器组件 - 使用 react-codemirror，现代深色主题
function CodeEditor({ code = '', readOnly = true }: CodeEditorProps) {
  // 默认代码
  const defaultCode = code || `# AlphaEvolve Generated Code
def steiner_tree(graph, terminals):
    """
    Compute Steiner Minimum Tree
    Using greedy algorithm approximation
    """
    import math

    n = len(graph.nodes)
    if n == 0 or len(terminals) == 0:
        return []

    # Initialize
    dist = {node: math.inf for node in graph.nodes}
    parent = {node: None for node in graph.nodes}
    terminal_set = set(terminals)

    # Prim's algorithm for MST
    current = terminals[0]
    dist[current] = 0
    visited = set()

    while len(visited) < n:
        visited.add(current)

        # Update distances
        for neighbor in graph.neighbors(current):
            weight = graph[current][neighbor].get('weight', 1)
            if dist[neighbor] > weight and neighbor not in visited:
                dist[neighbor] = weight
                parent[neighbor] = current

        # Find nearest unvisited node
        next_node = None
        min_dist = math.inf
        for node in graph.nodes:
            if node not in visited and dist[node] < min_dist:
                min_dist = dist[node]
                next_node = node

        if next_node is None:
            break
        current = next_node

    # Build tree edges
    edges = []
    for node in graph.nodes:
        if parent[node] is not None:
            edges.append((parent[node], node))

    return edges
`

  const [displayCode] = useState<string>(defaultCode)

  // 自定义深色主题 - 与新设计系统一致
  const customTheme = EditorView.theme({
    '&': {
      backgroundColor: 'rgba(15, 23, 42, 0.8)',
      color: '#e2e8f0',
      fontSize: '13px',
      borderRadius: '8px',
    },
    '.cm-content': {
      caretColor: '#818cf8',
      fontFamily: '"JetBrains Mono", "Fira Code", monospace',
      padding: '12px 0',
    },
    '.cm-gutters': {
      backgroundColor: 'rgba(30, 41, 59, 0.5)',
      color: '#64748b',
      border: 'none',
      borderRight: '1px solid rgba(148, 163, 184, 0.1)',
    },
    '.cm-activeLineGutter': {
      backgroundColor: 'rgba(139, 92, 246, 0.15)',
      color: '#a78bfa',
    },
    '.cm-activeLine': {
      backgroundColor: 'rgba(139, 92, 246, 0.08)',
    },
    '.cm-selectionBackground': {
      backgroundColor: 'rgba(139, 92, 246, 0.3) !important',
    },
    '&.cm-focused .cm-selectionBackground': {
      backgroundColor: 'rgba(139, 92, 246, 0.3) !important',
    },
    '.cm-cursor': {
      borderLeftColor: '#818cf8',
    },
    '.cm-matchingBracket': {
      backgroundColor: 'rgba(139, 92, 246, 0.3)',
      color: '#e2e8f0 !important',
    },
    '.cm-tooltip': {
      backgroundColor: '#1e293b',
      border: '1px solid rgba(148, 163, 184, 0.2)',
      borderRadius: '6px',
    },
    '.cm-tooltip-autocomplete': {
      '& > ul > li[aria-selected]': {
        backgroundColor: 'rgba(139, 92, 246, 0.3)',
      }
    }
  })

  return (
    <div className="rounded-lg overflow-hidden border border-slate-700/30">
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
          indentOnInput: true,
          bracketMatching: true,
          autocompletion: true,
        }}
      />
    </div>
  )
}

export default CodeEditor
