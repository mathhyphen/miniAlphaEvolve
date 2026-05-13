import { useState, useEffect } from 'react'
import { Download, Eye } from 'lucide-react'

// 存档数据类型
interface Archive {
  id: string
  generation: number
  score: number
  code: string
  timestamp: string
  parentId?: string
}

// 历史存档列表组件 - 显示所有历史版本，可点击查看详情，支持导出功能
function ArchiveList() {
  const [archives, setArchives] = useState<Archive[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // 模拟加载存档数据
  useEffect(() => {
    setArchives([
      { id: 'v12', generation: 12, score: 0.856, code: '# v12 code...', timestamp: '2026-04-01 10:30:00', parentId: 'v11' },
      { id: 'v11', generation: 11, score: 0.823, code: '# v11 code...', timestamp: '2026-04-01 10:25:00', parentId: 'v10' },
      { id: 'v10', generation: 10, score: 0.798, code: '# v10 code...', timestamp: '2026-04-01 10:20:00', parentId: 'v9' },
      { id: 'v9', generation: 9, score: 0.765, code: '# v9 code...', timestamp: '2026-04-01 10:15:00', parentId: 'v8' },
      { id: 'v8', generation: 8, score: 0.721, code: '# v8 code...', timestamp: '2026-04-01 10:10:00', parentId: 'v7' },
      { id: 'v7', generation: 7, score: 0.680, code: '# v7 code...', timestamp: '2026-04-01 10:05:00', parentId: 'v6' },
      { id: 'v6', generation: 6, score: 0.632, code: '# v6 code...', timestamp: '2026-04-01 10:00:00', parentId: 'v5' },
      { id: 'v5', generation: 5, score: 0.580, code: '# v5 code...', timestamp: '2026-04-01 09:55:00' },
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
    console.log('查看存档:', id)
  }

  return (
    <div className="code-panel">
      <div className="code-title">历史存档</div>
      <div className="archive-list">
        {archives.map((archive) => (
          <div
            key={archive.id}
            className={`archive-item ${selectedId === archive.id ? 'selected' : ''}`}
            onClick={() => handleView(archive.id)}
          >
            <div className="archive-info">
              <div className="archive-name">
                v{archive.generation}
                {selectedId === archive.id && ' (当前)'}
              </div>
              <div className="archive-meta">
                {archive.timestamp}
                {archive.parentId && ` · 源自 ${archive.parentId}`}
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="archive-score">{(archive.score * 100).toFixed(1)}%</span>
              <button
                className="btn-icon"
                onClick={(e) => handleExport(e, archive)}
                title="导出"
              >
                <Download size={14} />
              </button>
              <button
                className="btn-icon"
                onClick={(e) => {
                  e.stopPropagation()
                  handleView(archive.id)
                }}
                title="查看"
              >
                <Eye size={14} />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* 选中存档预览 */}
      {selectedId && (
        <div style={{
          marginTop: '12px',
          padding: '12px',
          background: 'rgba(0, 0, 0, 0.2)',
          borderRadius: '8px'
        }}>
          <div style={{ fontSize: '12px', color: '#a1a1aa', marginBottom: '8px' }}>
            代码预览:
          </div>
          <pre style={{
            fontSize: '11px',
            color: '#e0e0e0',
            maxHeight: '80px',
            overflow: 'auto',
            fontFamily: '"JetBrains Mono", monospace'
          }}>
            {archives.find(a => a.id === selectedId)?.code || '无'}
          </pre>
        </div>
      )}

      <style>{`
        .btn-icon {
          background: rgba(255, 255, 255, 0.1);
          border: none;
          border-radius: 6px;
          padding: 6px;
          cursor: pointer;
          color: #fff;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.2s;
        }
        .btn-icon:hover {
          background: rgba(255, 255, 255, 0.2);
        }
        .archive-item.selected {
          background: rgba(56, 239, 125, 0.15);
          border: 1px solid rgba(56, 239, 125, 0.3);
        }
      `}</style>
    </div>
  )
}

export default ArchiveList
