# Agent

## 项目结构

```txt
root/
├── backend/                  # Flask 应用核心层
│   ├── __init__.py           # Flask 应用工厂函数，蓝图注册
│   ├── config.py             # Flask 应用配置
│   ├── routes.py             # Web 路由定义 (API 端点)
│   └── embeddding.py         # 负责初始化 ChromaDB 和处理数据的摄入
├── agent/                    # 智能体系统核心层
│   ├── __init__.py           # 将 agent 目录变为 Python 包
│   ├── agents_core.py        # 智能体定义、状态、LangGraph 图构建、系统初始化
│   ├── tools.py              # 智能体可调用的工具函数
│   └── model_factory.py      # LLM 模型加载
├── frontend/                 # 前端资源目录
│   ├── templates/            # HTML 模板
│   │   └── index.html
│   └── static/               # 静态文件 (CSS, JS)
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── script.js
├── uploads/                  # 用户上传文件的临时存放目录
├── out/                      # AI 生成代码和日志的输出目录
└── run.py                    # 项目启动文件 (新的入口点)
```

