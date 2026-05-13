# AlphaEvolve Web UI 设计文档

## 概述

AlphaEvolve Web UI 是一个基于浏览器的图形界面，用于交互式地运行算法演进实验、监控演进过程、分析结果。

## 技术栈

- **前端：** React 18 + TypeScript + Vite
- **后端：** FastAPI (Python)
- **图表：** Recharts
- **样式：** 渐变现代风（Glassmorphism）

## 设计决策

### 布局：单页仪表盘

```
┌──────────────────────────────────────────────────────────────┐
│  Logo   [问题选择▼]  [Benchmark▼]     [开始] [停止]  ⚙️   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │Best Score│ │Generation│ │Candidates│ │Diversity │          │
│  │  0.847  │ │   42    │ │   128   │ │   0.73  │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           演进曲线图表 (分数 vs Generation)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │    代码对比面板      │  │      历史存档列表           │   │
│  │  [原始]  vs  [当前]  │  │  • gen-1  score: 0.72     │   │
│  │                      │  │  • gen-5  score: 0.81     │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 视觉风格：渐变现代风

- 背景：渐变色 `#667eea → #764ba2`
- 卡片：玻璃态 `rgba(255,255,255,0.15)` + `backdrop-filter: blur`
- 文字：白色/浅色系

## 功能模块

### 1. 自定义问题接入

- 用户输入算法代码（文本编辑器）
- 用户定义评分函数 `evaluate(code) -> float`
- 一键启动演进

### 2. 内置 Benchmark 库

- MST (Minimum Spanning Tree)
- Steiner Tree
- 排序算法
- 矩阵乘法

### 3. 演进可视化

- 实时分数曲线（Recharts LineChart）
- 多样性指标仪表
- 当前最佳代码展示

### 4. 代码对比分析

- Side-by-side diff 视图
- 语法高亮
- 差异行标记

### 5. 历史存档管理

- 保存演进结果到 JSON
- 导出/导入功能
- 历史记录列表

## 后端 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/problems` | GET | 获取问题列表 |
| `/api/benchmarks` | GET | 获取 Benchmark 列表 |
| `/api/evolution/start` | POST | 启动演进 |
| `/api/evolution/stop` | POST | 停止演进 |
| `/api/evolution/status` | GET | 获取演进状态 |
| `/api/evolution/results` | GET | 获取演进结果 |
| `/api/archive` | GET | 获取存档列表 |
| `/api/compare` | POST | 对比两个版本 |

## 项目结构

```
alphaevolve_webui/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── api/
│   │   └── App.tsx
│   └── package.json
├── backend/
│   ├── main.py
│   ├── api/
│   ├── evolution/
│   └── requirements.txt
└── README.md
```

## 实现顺序

1. 后端 API 框架搭建
2. 问题接口标准化
3. 添加更多 Benchmark
4. 演进结果可视化（后端支持）
5. React 前端开发
6. WebSocket 实时更新
7. 代码对比功能
8. 历史存档功能
