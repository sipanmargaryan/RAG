# RAG

This FastAPI application processes PDF documents and finds passages relevant to a
natural-language query. It provides document ingestion, retrieval, and answer
generation for a retrieval-augmented generation (RAG) system, returning an answer
with citations and source document metadata.

## How it works

**PDF upload:** The API extracts text from an uploaded PDF, using OCR for pages
that need it. It splits the text into overlapping chunks and generates normalized
MiniLM embeddings. The original PDF is stored in MinIO, document metadata in
PostgreSQL, and the chunks and their embeddings in Qdrant.

**Search:** OpenAI, Gemini, or Claude generates four variations of the user's query. The
original query and these variations form five queries, which are embedded and searched
against Qdrant in one batch. Each query retrieves up to five chunks, producing
up to 25 candidates. Duplicate chunk IDs are removed, and a CrossEncoder reranks
the unique candidates against the original query. The five best chunks (or fewer
if fewer candidates exist) and original question are sent to the answer provider.
The provider is instructed to use only those passages and acknowledge insufficient
evidence. The API returns the generated answer with inline citations such as [1]
and the cited source chunks, including their metadata and reranker scores.

Search requires the selected provider's API key in `.env`. OCR requires Tesseract.
Embedding and reranking models load on first use and may download model files. Run the database
and Qdrant migrations before uploading or searching documents.

Set both `QUERY_EXPANSION_PROVIDER` and `ANSWER_PROVIDER` explicitly to `openai`,
`gemini`, or `claude`. OpenAI uses `OPENAI_API_KEY`, `OPENAI_MODEL`, and
`OPENAI_TIMEOUT_SECONDS`. To use Gemini for both stages:

```env
QUERY_EXPANSION_PROVIDER=gemini
ANSWER_PROVIDER=gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_TIMEOUT_SECONDS=30
GEMINI_MAX_RETRIES=2
```

Gemini uses the Interactions API with [structured JSON output](https://ai.google.dev/gemini-api/docs/structured-output).
All providers return the original query plus four distinct variations. Failed
requests, incomplete responses, or invalid variations return HTTP 503 before
embedding or vector search. Only the selected provider is called.

Gemini overload responses (HTTP 503) are retried up to `GEMINI_MAX_RETRIES` times
with exponential backoff and jitter, within the overall `GEMINI_TIMEOUT_SECONDS`
deadline. Requests use low thinking for query rewriting. Quota/rate limits (429),
access errors, unavailable models, and timeouts return distinct messages. A free
tier key is still subject to project eligibility, quotas, and model availability;
temporary model overload does not by itself indicate that billing is required.

The two provider settings are independent and required; neither falls back to
the other. Answer generation always uses the selected provider's configured
model: `OPENAI_MODEL`, `GEMINI_MODEL`, or `CLAUDE_MODEL`.
`ANSWER_MAX_OUTPUT_TOKENS` defaults to 2048. The chosen provider's API key and
timeout settings apply; Gemini answer requests also use the overload retry policy.
With matching documents, a search makes one expansion call and one answer call
(plus any retries), which both count toward provider usage limits.

If no chunks are found, answer generation makes no provider call and returns an
insufficient-information message with empty sources. Incomplete, refused, invalid,
or failed provider responses return HTTP 503; generated source references are
validated against the supplied chunks before returning the answer.

To use Claude for expansion and answer generation:

```env
QUERY_EXPANSION_PROVIDER=claude
ANSWER_PROVIDER=claude
CLAUDE_API_KEY=your-key
CLAUDE_MODEL=claude-haiku-5-5
CLAUDE_TIMEOUT_SECONDS=60
CLAUDE_MAX_RETRIES=2
```

Claude uses Anthropic's Messages API and [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
`CLAUDE_API_VERSION` defaults to `2026-10-21`. Set `CLAUDE_WORKSPACE_ID` when
your API key requires an explicit workspace; workspace-scoped keys may omit it.
Unsupported JSON schema constraints are adapted for Claude's request format,
while full validation still checks four distinct variations and valid answer
citations locally. Truncated, refused, or malformed outputs are rejected.
Temporary HTTP 502/503/504/529 errors are retried up to `CLAUDE_MAX_RETRIES`
times within `CLAUDE_TIMEOUT_SECONDS`; quota and authentication errors return
clear messages without switching providers.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/rag/pdf` | Upload and process a PDF using multipart form field `file`. |
| `POST` | `/rag/search` | Get a grounded answer and cited sources with a JSON body such as `{"text": "How are PDFs stored?"}`. |
| `GET` | `/health` | Check that the API is running; database connectivity is not checked. |

Responses contain a `message` and optional `data`. Search now returns an object
with `data.answer` and `data.sources`, replacing the previous flat chunk list.
Each source includes `source_id` (matching inline citations), text, score, and PDF
metadata. Interactive API
documentation is available at `/docs`, and the OpenAPI schema at `/openapi.json`.

## Commands

```bash
make help                      # List all Make commands
make build                     # Build the API image
make up                        # Start all services
make down                      # Stop containers and keep stored data
make logs                      # Follow container logs
make ps                        # Show container status
make config                    # Validate Docker Compose configuration
make lint                      # Run Ruff lint checks
make format-check              # Check Python formatting
make format                    # Format source files on the host using the container
make test                      # Run unit tests without a database; accepts ARGS="..."
make integration               # Run Testcontainers tests on the host (requires uv and Docker)
make check                     # Validate Compose and run tox; accepts ARGS="..."
make shell                     # Open a shell in the API container
make hooks                     # Enable Git hooks
make migration ARGS='-m "describe schema change"'  # Generate a database migration
make migrate                   # Apply database migrations
make migration-check           # Check database schema against ORM models
make qdrant-migrate             # Create or migrate the Qdrant collection
make qdrant-migration-check     # Validate the Qdrant collection schema
```

## Git hooks

```bash
make hooks                     # Enable Git hooks from .githooks for this clone
```

## Database migrations

```bash
make migration ARGS='-m "describe schema change"'  # Generate a migration
make migrate                                     # Apply Alembic migrations to PostgreSQL
make migration-check                             # Check schema against ORM models
```

## Qdrant migrations

```bash
make up                        # Start services before running Qdrant migrations
make qdrant-migrate             # Create or migrate the Qdrant collection
make qdrant-migration-check     # Validate the collection schema
```
