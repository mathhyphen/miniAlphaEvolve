import { useState, useEffect } from 'react'
import { Download, Eye, Clock, GitBranch, ChevronRight } from 'lucide-react'

// 存档数据类型
interface Archive {
  id: string
  generation: number
  score: number
  code: string
  timestamp: string
  parentId?: string
}

// 历史存档列表组件 - 现代表格设计
function ArchiveList() {
  const [archives, setArchives] = useState<Archive[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  // 模拟加载存档数据
  useEffect(() => {
    setArchives([
      { id: 'v12', generation: 12, score: 0.856, code: '# v12 code...\ndef steiner_tree_optimized():\n    pass', timestamp: '2026-04-01 10:30:00', parentId: 'v11' },
      { id: 'v11', generation: 11, score: 0.823, code: '# v11 code...', timestamp: '2026-04-01 10:25:00', parentId: 'v10' },
      { id: 'v10', generation: 10, score: 0.798, code: '# v10 code...', timestamp: '2026-04-01 10:20:00', parentId: 'v9' },
      { id: 'v9', generation: 9, score: 0.765, code: '# v9 code...', timestamp: '2026-04-01 10:15:00', parentId: 'v8' },
      { id: 'v8', generation: 8, score: 0.721, code: '# v8 code...', timestamp: '2026-04-01 10:10:00', parentId: 'v7' },
      { id: 'v7', generation: 7, score: 0.680, code: '# v7 code...', timestamp: '2026-04-01 10:05:00', parentId: 'v6' },
    ])
  }, [])

  // 导出存档
  const handleExport = (e: React.MouseEvent, archive: Archive) => {
    e.stopPropagation()
    try {
      const blob = new Blob([archive.code], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `alphaevolve_v${archive.generation}_${archive.id}.py`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('导出失败:', error)
    }
  }

  // 查看存档详情
  const handleView = (id: string) => {
    setSelectedId(id)
  }

  const selectedArchive = archives.find(a => a.id === selectedId)

  return (
    <div className="h-full flex flex-col">
      {/* 标题 */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30">
            <GitBranch className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-100">Version Archive</h3>
            <p className="text-xs text-slate-400">{archives.length} versions saved</p>
          </div>
        </div>
      </div>

      {/* 存档列表 */}
      <div className="flex-1 overflow-auto space-y-2 min-h-0">
        {archives.map((archive) => (
          <div
            key={archive.id}
            className={`
              relative p-4 rounded-lg border cursor-pointer transition-all duration-200
              ${selectedId === archive.id
                ? 'bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border-emerald-500/30'
                : hoveredId === archive.id
                  ? 'bg-slate-800/50 border-slate-600/30'
                  : 'bg-slate-800/30 border-slate-700/20 hover:border-slate-600/40'
              }
            `}
            onClick={() => handleView(archive.id)}
            onMouseEnter={() => setHoveredId(archive.id)}
            onMouseLeave={() => setHoveredId(null)}
          >
            <div className="flex items-center justify-between">
              {/* 左侧信息 */}
              <div className="flex items-center gap-4">
                {/* 版本号 */}
                <div className={`
                  w-12 h-12 rounded-xl flex flex-col items-center justify-center font-bold
                  ${archive.id === 'v12'
                    ? 'bg-gradient-to-br from-emerald-400 to-cyan-400 text-slate-900'
                    : 'bg-slate-700/50 text-slate-300'
                  }
                `}>
                  <span className="text-xs opacity-75">v</span>
                  <span className="text-lg leading-none">{archive.generation}</span>
                </div>

                {/* 详情 */}
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-slate-200">
                      Version {archive.generation}
                    </span>
                    {archive.id === 'v12' && (
                      <span className="badge badge-success text-[10px]">Current</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-slate-400">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {archive.timestamp.split(' ')[1]}
                    </span>
                    {archive.parentId && (
                      <span className="flex items-center gap-1">
                        <GitBranch className="w-3 h-3" />
                        {archive.parentId}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* 右侧信息 */}
              <div className="flex items-center gap-3">
                {/* 分数 */}
                <div className="text-right">
                  <div className={`
                    font-bold text-lg
                    ${archive.score >= 0.8
                      ? 'text-emerald-400'
                      : archive.score >= 0.6
                        ? 'text-amber-400'
                        : 'text-slate-400'
                    }
                  `}>
                    {(archive.score * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">Score</div>
                </div>

                {/* 操作按钮 */}
                <div className="flex items-center gap-1">
                  <button
                    className="btn-icon"
                    onClick={(e) => handleExport(e, archive)}
                    title="Export"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  <button
                    className="btn-icon"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleView(archive.id)
                    }}
                    title="View"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                  <ChevronRight className="w-4 h-4 text-slate-500" />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 选中存档预览 */}
      {selectedArchive && (
        <div className="mt-4 p-4 rounded-lg bg-slate-900/50 border border-slate-700/30 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-slate-300">Code Preview</span>
            <span className="badge badge-info text-xs">v{selectedArchive.generation}</span>
          </div>
          <pre className="text-xs text-slate-400 overflow-auto max-h-24 leading-relaxed">
            <code>{selectedArchive.code}</code>
          </pre>
        </div>
      )}
    </div>
  )
}

export default ArchiveList
