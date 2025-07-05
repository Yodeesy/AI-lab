# Agent

## Project Structure

```txt
root/
├── backend/                  # Flask application core layer
│   ├── __init__.py           # Flask application factory function, blueprint registration
│   ├── config.py             # Flask application configuration
│   ├── routes.py             # Web route definitions (API endpoints)
│   └── embeddding.py         # Responsible for initializing ChromaDB and handling data ingestion
├── agent/                    # Agent system core layer
│   ├── __init__.py           # Make the agent directory a Python package
│   ├── agents_core.py        # Agent definition, state, LangGraph construction, system initialization
│   ├── tools.py              # Tool functions callable by the agent
│   └── model_factory.py      # LLM model loading
├── frontend/                 # Frontend resource directory
│   ├── templates/            # HTML templates
│   │   └── index.html
│   └── static/               # Static files (CSS, JS)
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── script.js
├── uploads/                  # Temporary storage directory for user-uploaded files
├── out/                      # Output directory for AI-generated code and logs
├── run.py                    # Project startup file (new entry point)
└── requirements.txt          # Dependencies
```
---
## Running the Project
- Run run.py
- Open http://127.0.0.1:5000/
