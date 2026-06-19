# EcoHome Energy Advisor

A LangGraph agent that helps a household decide *when* and *how* to use energy
(EV charging, HVAC, appliances, home battery, solar) to lower electricity cost
and emissions. It reasons over weather forecasts, time-of-use prices, the
household's own usage/solar history, and a small knowledge base of energy tips.

## How it works

The agent is a `create_react_agent` (LangGraph) backed by `gpt-4o-mini` via the
Vocareum OpenAI endpoint. It has seven tools:

- `get_weather_forecast` – hourly temperature, condition and solar irradiance
- `get_electricity_prices` – time-of-use rates (off-peak / mid-peak / peak)
- `query_energy_usage` – consumption history from the local database
- `query_solar_generation` – solar production history
- `get_recent_energy_summary` – recent usage + generation rollup
- `search_energy_tips` – RAG search over the knowledge base (ChromaDB)
- `calculate_energy_savings` – savings from a usage change

The weather and price tools are seeded mocks (seeded by location / date), so the
same input always returns the same data.

## Project structure

```
ecohome_solution/
├── agent.py                 # Agent class (LLM + graph + invoke)
├── tools.py                 # the seven agent tools
├── models/energy.py         # SQLAlchemy models + DatabaseManager
├── data/
│   ├── documents/           # 7 energy-tip text files (RAG knowledge base)
│   ├── energy_data.db        # SQLite DB, created by 01_db_setup
│   └── vectorstore/          # Chroma store, created by 02_rag_setup
├── 01_db_setup.ipynb        # build the database and sample data
├── 02_rag_setup.ipynb       # build the vector store and test retrieval
├── 03_run_and_evaluate.ipynb# run the agent on 10 cases and evaluate
├── requirements.txt
└── README.md
```

## Setup

1. Install the dependencies (Python 3.11):

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in `ecohome_solution/` with your Vocareum key:

   ```
   VOCAREUM_API_KEY=your_key_here
   ```

3. Run the notebooks in order: `01_db_setup` → `02_rag_setup` →
   `03_run_and_evaluate`. The first two create `data/energy_data.db` and
   `data/vectorstore/`; the third runs and scores the agent.

## Evaluation

`03_run_and_evaluate.ipynb` runs the agent on 10 test cases and reports two
things: an LLM-judge score for each answer (accuracy, relevance, completeness,
usefulness) and a deterministic tool-usage score (were the expected tools
actually called). The report aggregates both and lists strengths, weaknesses
and recommendations.
