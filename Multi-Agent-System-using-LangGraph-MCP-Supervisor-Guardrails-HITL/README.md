# TripMate AI

TripMate AI is a multi-agent travel-planning application built with FastAPI, LangGraph, Groq, MCP, and PostgreSQL checkpointing. It accepts a natural-language travel request, validates and routes it, gathers specialist information, creates a draft itinerary, pauses for human review, and then produces a final plan.

The repository demonstrates:

- Supervisor-based agent routing
- LLM input guardrails
- MCP tool integration over HTTP and stdio
- Human-in-the-loop approval with resumable graph state
- PostgreSQL-backed LangGraph checkpoints
- A browser frontend for drafting, reviewing, copying, and downloading plans

## Architecture

```text
Browser
  |
  |  POST /api/travel
  v
FastAPI (app.py)
  |
  v
LangGraph StateGraph (backend.py)
  |
  +--> Supervisor + input guardrail
  |       |
  |       +--> block unrelated or harmful requests
  |       +--> select specialist agents and extract constraints
  |
  +--> Flight Agent ------> AviationStack MCP via uvx/stdio
  +--> Hotel Agent -------> Tavily MCP via streamable HTTP
  +--> Weather Agent -----> Local Weather MCP via Python/stdio
  +--> Budget Agent -------> Groq reasoning over collected context
  +--> Itinerary Agent ----> Groq draft generation
  |
  +--> Human approval interrupt
  |       |
  |       +--> approve or provide revision feedback
  |
  +--> Final Agent --------> Groq final response
  |
  +--> PostgreSQL ---------- LangGraph checkpoint persistence
```

### Components

| Component | Responsibility |
| --- | --- |
| `app.py` | FastAPI application, HTML route, health endpoint, travel and approval APIs |
| `backend.py` | LangGraph state, agents, routing, HITL interrupt, PostgreSQL checkpointer, API wrappers |
| `mcp_client.py` | Groq destination extraction and Tavily, AviationStack, and weather MCP clients |
| `custom_weather_mcp_server.py` | Local FastMCP server exposing OpenWeather tools |
| `templates/index.html` | Browser interface |
| `static/script.js` | API calls, approval state, Markdown rendering, copy and PDF actions |
| `static/style.css` | Frontend styling |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile` | Container image definition |

## LangGraph design

The graph uses a `TravelState` TypedDict containing the user query, guardrail result, selected agents, extracted constraints, specialist results, draft itinerary, approval state, final response, and an approximate `llm_calls` count.

### Nodes

1. `supervisor`: runs the input guardrail and, when allowed, selects specialists and constraints.
2. `guardrail_blocked`: returns the guardrail explanation without running travel tools.
3. `flight_agent`: loads airport and airline data from AviationStack MCP, then asks Groq for flight guidance.
4. `hotel_agent`: searches Tavily MCP for accommodation recommendations.
5. `weather_agent`: extracts the destination with Groq and loads current weather plus forecast from the local weather MCP server.
6. `budget_agent`: assesses cost categories, risks, savings, and feasibility using collected context.
7. `itinerary_agent`: combines available specialist results into a reviewable draft.
8. `human_approval`: pauses with LangGraph `interrupt()` and exposes the draft for review.
9. `final_agent`: incorporates approval or revision feedback and generates the final response.

### Routing

The supervisor returns strict JSON with `selected_agents`, `trip_constraints`, and `reasoning`. The application filters names against the known ordered list and always includes `itinerary_agent`.

Selected specialists execute sequentially in this order:

```text
flight_agent -> hotel_agent -> weather_agent -> budget_agent -> itinerary_agent
```

Agents not selected are skipped. The itinerary agent receives whichever specialist results exist. If guardrail or supervisor parsing fails, the current implementation uses a fallback: guardrail parsing fails open, while supervisor parsing selects the full workflow.

## End-to-end workflows

### New travel plan

1. The browser sends `POST /api/travel`.
2. FastAPI trims and validates the message.
3. `run_travel_agent()` creates a `thread_id` when needed.
4. LangGraph persists state using that thread ID.
5. The guardrail checks that the request is travel-related and not harmful or illegal.
6. The supervisor selects agents and extracts constraints.
7. Selected agents gather live or inferred information.
8. The itinerary agent creates a draft.
9. The graph pauses at `human_approval`.
10. The API returns the draft with `requires_approval: true`.

### Approve a draft

1. Send `approved: true` to `POST /api/travel/approve` with the same `thread_id`.
2. `resume_travel_agent()` resumes the persisted interrupt using `Command(resume=...)`.
3. The final agent polishes the approved draft.
4. The API returns the final response with `requires_approval: false`.

### Request revisions

1. Send `approved: false` and non-empty `feedback` with the same thread ID.
2. The graph resumes and the final agent receives the feedback.
3. The final agent generates a revised final plan.

### Guardrail-blocked request

An unrelated or harmful request routes directly to `guardrail_blocked`. No specialist workflow is run.

## MCP integrations

`mcp_client.py` loads only the MCP server needed by each specialist, so an unavailable integration does not necessarily prevent unrelated work.

### Tavily MCP

- Transport: streamable HTTP
- Endpoint: `https://mcp.tavily.com/mcp/`
- Tool: `tavily_search`
- Consumer: hotel agent
- Credential: `TAVILY_API_KEY`

### AviationStack MCP

- Transport: stdio
- Launcher: `uvx aviationstack-mcp`
- Tools: `list_airports` and `list_airlines`
- Consumer: flight agent
- Credential: `AVIATION_STACK_API_KEY` or `AVIATIONSTACK_API_KEY`
- Requirement: `uvx` must be available on `PATH`

### Local weather MCP

- Transport: stdio
- Launcher: the active Python interpreter running `custom_weather_mcp_server.py`
- Tools: `get_current_weather` and `get_forecast`
- Consumer: weather agent
- Credential: `OPENWEATHER_API_KEY`

The local weather server calls OpenWeather with a 20-second timeout and returns compact current-weather and forecast dictionaries.

### MCP diagnostic helper

`mcp_client.py` includes `get_all_tools()`, which independently loads each configured server and prints available tools or failure reasons. Use it to diagnose credentials, network access, `uvx`, or MCP package issues.

## API reference

### `GET /`

Serves the TripMate HTML frontend.

### `GET /health`

Returns a basic service status:

```json
{
  "status": "ok",
  "message": "TripMate AI API is running",
  "features": ["supervisor_agent", "input_guardrail", "human_in_the_loop"]
}
```

This is an application health check. It does not make a Groq, PostgreSQL, or MCP request.

### `POST /api/travel`

Request:

```json
{
  "message": "Plan a 7-day Japan trip from Bangladesh under 200000 BDT",
  "thread_id": "optional-existing-thread-id"
}
```

`message` is required. `thread_id` is optional for a new workflow and must be reused to continue persisted state.

The response contains:

```json
{
  "success": true,
  "thread_id": "user_<uuid>",
  "answer": "draft or final text",
  "requires_approval": true,
  "approval_request": "Please review the generated draft itinerary...",
  "itinerary": "draft itinerary text",
  "flight_results": "...",
  "hotel_results": "...",
  "weather_results": "...",
  "budget_results": "...",
  "selected_agents": ["flight_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {},
  "supervisor_reasoning": "...",
  "guardrail_allowed": true,
  "guardrail_reason": "",
  "approved": false,
  "human_feedback": "",
  "llm_calls": 0
}
```

The serialized result keeps these fields stable even when a value is empty.

### `POST /api/travel/approve`

Approve:

```json
{
  "thread_id": "user_<uuid>",
  "approved": true,
  "feedback": ""
}
```

Request revisions:

```json
{
  "thread_id": "user_<uuid>",
  "approved": false,
  "feedback": "Reduce hotel cost and add one free day."
}
```

Invalid empty messages return HTTP 400, Pydantic validation failures return HTTP 422, and unexpected workflow errors return HTTP 500 with an `error` field.

## Configuration

Create `.env` in the project root and never commit real secrets:

```dotenv
DATABASE_URL=postgresql://user:password@host:5432/database
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.6-27b
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
DEFAULT_ORIGIN_IATA=DAC

# Optional LangSmith tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=tripmate-ai
LANGSMITH_END_POINT=https://api.smith.langchain.com
```

Notes:

- `DATABASE_URL` and `GROQ_API_KEY` are required when `backend.py` is imported.
- `GROQ_MODEL` defaults to `qwen/qwen3.6-27b`, but can be set to any model ID available to the configured Groq account.
- `DATABASE_URL` receives `sslmode=require` automatically when it has no `sslmode` parameter.
- Tavily, AviationStack, and OpenWeather keys are required when their tools are reached.
- `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` are set from the installed `certifi` bundle at startup.
- There is no committed `.env.example`; use the template above.

## Installation and local run

### Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the API:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

For development reload:

```powershell
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

If the terminal is outside the repository directory, use `--app-dir`:

```powershell
python -m uvicorn app:app --app-dir "C:\path\to\Multi-Agent-System-using-LangGraph-MCP-Supervisor-Guardrails-HITL" --host 127.0.0.1 --port 8000
```

### Weather MCP server

The application launches this server automatically over stdio. To inspect it independently:

```powershell
python custom_weather_mcp_server.py
```

It communicates using MCP stdio and is not a browser HTTP server.

### Docker

The `Dockerfile` installs requirements, copies the application, exposes port 8000, and runs Uvicorn on `0.0.0.0`:

```powershell
docker build -t tripmate-ai .
docker run --rm -p 8000:8000 --env-file .env tripmate-ai
```

The container needs network access to Groq, PostgreSQL, Tavily, AviationStack, and OpenWeather as applicable.

## Frontend workflow

The browser UI:

1. Stores the current `thread_id` in `localStorage`.
2. Sends the prompt and existing thread ID to `/api/travel`.
3. Displays supervisor reasoning, guardrail status, and selected agents.
4. Shows the itinerary as a draft when approval is required.
5. Enables approval or revision feedback.
6. Sends the decision to `/api/travel/approve`.
7. Renders the final Markdown response.
8. Supports copying and PDF download.

The frontend loads `marked` and `html2pdf.js` from CDNs, so internet access is needed for those optional browser features unless assets are vendored locally.

## Persistence and concurrency

The application opens PostgreSQL when `backend.py` loads, creates a `PostgresSaver`, runs `checkpointer.setup()`, and compiles the graph with that checkpointer.

Each workflow uses a UUID-backed ID such as `user_<uuid>`. The same ID is required to resume the approval interrupt. Do not reuse one thread ID for unrelated trips.

FastAPI routes are asynchronous, while current graph helpers invoke synchronous LangGraph and MCP convenience calls. `nest_asyncio` supports these existing nested calls. For production workloads, move blocking graph execution to a worker or redesign around fully async execution.

## Troubleshooting

### `model_not_found` from Groq

The model is unavailable to the API key. List models available to the account and set `GROQ_MODEL` to one of those IDs:

```powershell
python -c "import os; from dotenv import load_dotenv; from groq import Groq; load_dotenv(); print('\\n'.join(sorted(model.id for model in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data)))"
```

### Missing Groq or database credentials

Confirm `.env` is in the application directory and contains non-empty `GROQ_API_KEY` and `DATABASE_URL` values. PostgreSQL must be reachable because HITL state is persisted with `PostgresSaver`.

### `uvx was not found`

Install `uv`, reopen PowerShell, and confirm:

```powershell
uvx --version
```

### MCP credential or network failures

Hotel, weather, and flight paths have fallback behavior for some external failures. These fallbacks do not replace live data; final answers should label estimates and unavailable data accordingly.

### Port 8000 is in use

Use another port:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

### Stale server code

Stop old Uvicorn processes and start one fresh process. Running processes keep imported model and graph objects in memory.

## Validation commands

```powershell
python -m compileall -q app.py backend.py mcp_client.py custom_weather_mcp_server.py
python -c "import app; print('APP_IMPORT_OK')"
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing
```

There are currently no automated tests. A useful manual acceptance flow is: call `/health`, submit a travel request, confirm a draft and `thread_id`, approve or revise with that same ID, and confirm a final response.

## Security and production considerations

- Keep `.env`, API keys, database credentials, and generated logs out of version control.
- Treat generated travel information and prices as advisory; verify bookings, visa requirements, weather, and policies with authoritative providers.
- Add authentication, authorization, request limits, rate limiting, structured logging, and production timeouts before public exposure.
- Sanitize public error responses instead of returning raw development exceptions.
- Use a PostgreSQL pool or async database strategy for higher concurrency.
- Review dependencies regularly, especially MCP adapters and external API clients.
- Add automated tests for guardrail routing, selected-agent routing, checkpoint resume, approval validation, and MCP failure fallbacks.

## License

See [LICENSE](LICENSE) for the repository license.
