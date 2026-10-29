# RAG

## Swagger UI

Start the API with `uv run uvicorn app.main:app --reload`, then open
[Swagger UI](http://localhost:8000/docs). The OpenAPI schema is available at
`http://localhost:8000/openapi.json` and can also be imported into Postman.

Expand **POST /rag/pdf**, select **Try it out**, choose a PDF in the `file`
field, and click **Execute**. Swagger displays the HTTP status and response
body, with documented examples for success and validation errors. Successful
processing returns a confirmation without exposing the chunks. Use
**GET /health** to check API availability.

## PDF extraction and OCR

`app.helpers.utils._read_pdf(path)` extracts embedded text and uses Tesseract
OCR at 300 DPI for pages without text. It returns `PDFPage` records with
`page_number` (one-based), `text`, and `used_ocr`, rather than plain strings.
Empty pages are skipped without renumbering subsequent pages. A PDF with no
recognized text raises `ValueError`; invalid PDFs and OCR failures raise errors.

The API Docker image includes Tesseract with English language data. For local
development on Debian/Ubuntu, install `tesseract-ocr tesseract-ocr-eng` with apt
and run `uv sync --locked` for the Python dependencies. Other languages require
their Tesseract language packages, then pass e.g. `ocr_language="eng+fra"`.

Use `force_ocr=True` for mixed image/text pages or scans whose existing text
layer is incomplete. Automatic fallback only detects pages with no extracted
text. OCR has a configurable per-page timeout (`ocr_timeout=60` seconds).
Call this synchronous helper in a worker thread from async endpoints.

## Prepare PDF text for embedding

```python
from app.helpers.processing import chunk_texts
from app.helpers.utils import _read_pdf

pages = _read_pdf("document.pdf")
chunks = chunk_texts([page.text for page in pages])
```

The MiniLM tokenizer splits each input separately into chunks of up to 254
content tokens with 32 tokens of overlap, reserving two tokens for the model's
special tokens. The model loads on first use and may need to
download its files. This helper returns text chunks, not embeddings or token IDs.

Upload a PDF to the API using multipart form field `file`:

```bash
curl -X POST http://localhost:8000/rag/pdf -F 'file=@document.pdf'
```

The endpoint extracts text and creates chunks internally, then returns
`{"message": "PDF processed successfully", "data": null}`. Chunks are neither
included in the response. The original PDF is saved to MinIO after parsing,
chunking, and embedding succeed. Chunks are persisted in Qdrant and PDF metadata
is committed in PostgreSQL before the endpoint returns success. Parsing, OCR,
tokenization, embedding, and MinIO access run in a worker thread; PostgreSQL
and Qdrant calls are asynchronous. Temporary files are always removed. Invalid
or textless PDFs return HTTP 422; storage or embedding failures return HTTP 503.

Successful endpoints use the shared `APIResponse[Data]` model from
`app.helpers.responses`: a `message` and optional typed `data`. Endpoints with
no payload use `APIResponse[None]`. Application errors, HTTP errors, request
validation failures, and unexpected server errors use the same envelope via
`app.helpers.exceptions`. Validation details are in `data.errors`; unexpected
errors return a generic message and are logged on the server.

## MinIO PDF storage

`app.helpers.minio` provides a shared client, FastAPI dependency, and PDF upload
helper. PDFs are stored in the configured bucket under unique
`processed/<uuid>.pdf` keys with `application/pdf` content type. The bucket is
created on the first successful upload if needed. Stored PDFs preserve the
original bytes; OCR text and chunks are not written back into the PDF.

Docker Compose builds MinIO from the official source pinned to
`RELEASE.2025-10-15T17-29-55Z` using `Dockerfile.minio`, because the old public
container image is no longer available. The first build downloads Go modules
and compiles the server; subsequent builds use Docker's cache.

Docker Compose includes MinIO with a persistent volume. Run
`docker compose up -d --build` and open its console at
`http://localhost:9001`. The local development credentials default to
`minioadmin` / `minioadmin`; set `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` for
your deployment. Python configuration is listed in `.env.example`.

When running the API on the host, use `MINIO_ENDPOINT=localhost:9000`.
Compose configures the API to use `minio:9000`. Set `MINIO_SECURE=true` for a
TLS endpoint; the endpoint itself contains no `http://` or `https://` prefix.
The shared HTTP pool is released at application shutdown.

## Qdrant chunk storage

Uploads generate normalized 384-dimensional embeddings using
`sentence-transformers/all-MiniLM-L6-v2` and upsert them into
`pdf_chunks_minilm_l6_v2` with cosine distance. Create the collection explicitly
before uploading PDFs:

```bash
make qdrant-migrate
make qdrant-migration-check
```

These commands run inside the running API container so they use the Docker
network's Qdrant address. For a host-accessible Qdrant installation, use
`uv run --locked python -m app.migrations.qdrant` with the appropriate
`QDRANT_HOST` and `QDRANT_PORT`; add `--check` for validation without changes.

The migration creates the collection and records `schema_revision=0001` in
its metadata. It can adopt an existing compatible collection without changing
points, and re-running it is safe. Unknown revisions and incompatible vector
settings are rejected. Uploads perform only point writes; they do not check,
create, or modify collection schemas. A missing collection causes ingestion
to fail with HTTP 503 until the migration is applied.

Each point contains the chunk `text`, zero-based `chunk_index`,
PostgreSQL `document_id`, MinIO `bucket_name`/`object_key`, original filename,
and embedding model name. Point IDs are deterministic per object and chunk,
so retries of the same upsert do not duplicate points.

`QDRANT_COLLECTION_NAME` and `QDRANT_BATCH_SIZE` configure the collection and
write batch size. `EMBEDDING_MODEL_NAME`, `EMBEDDING_VECTOR_SIZE`, and
`EMBEDDING_BATCH_SIZE` configure inference. Use a new collection when changing
the model or vector size, and apply the migration before accepting uploads.
Dimensions, distance, and model identity are checked by the migration command.
The splitter's
254-token content budget is chosen for MiniLM; adjust it if changing models.
The model is cached per application worker; the first request may download
weights. Chunking and embedding currently have separate cached model instances.

Qdrant writes use `wait=True`. On a partial write failure the service rolls
back metadata and attempts to delete points belonging to that MinIO object.
It removes the PDF after confirmed vector cleanup; if cleanup fails, artifacts
are preserved and logged for reconciliation. A failed PostgreSQL commit also
preserves the PDF and vectors because its outcome may be uncertain. These
three services do not share an atomic transaction.

## Search PDF chunks

Send text to `POST /rag/search`:

```bash
curl -X POST http://localhost:8000/rag/search \
  -H 'Content-Type: application/json' \
  -d '{"text": "How are PDFs stored?"}'
```

The endpoint embeds the text with the same MiniLM model used for ingestion and
searches the configured Qdrant collection. It returns at most five chunks in
descending cosine similarity order, inside the shared response envelope:
`{"message": "Search completed", "data": [...]}`. Each result contains `text`,
`score`, `document_id`, `chunk_index`, `original_filename`, `bucket_name`, and
`object_key`. Vectors are not returned. An empty collection returns `data: []`;
blank or missing text returns HTTP 422. Apply `make qdrant-migrate` before
searching. This endpoint reads existing chunks and does not create collections.

## Git hooks

After cloning, run this command once from the repository root to enable the hooks in `.githooks` for your local clone:

```bash
make hooks
```

This runs `git config --local core.hooksPath .githooks`.

## Database dependencies

Processed PDF metadata lives in `pdf_documents`. Each row stores its MinIO
`bucket_name` and `object_key`, the original filename, page and chunk counts,
and creation/update timestamps. The bucket/key pair is unique. The PDF bytes
stay in MinIO; PostgreSQL stores the reference.

The upload endpoint commits the metadata before returning success. A metadata
flush failure rolls back and attempts to remove the uploaded object. A commit
failure preserves the object because the transaction may have reached the
server before the connection failed. Cleanup failures are logged; MinIO and
PostgreSQL do not share a transaction, so interrupted requests can leave
objects requiring reconciliation.

Endpoints receive database resources through FastAPI dependencies:

```python
from app.db.postgres import AsyncSessionDep
from app.db.qdrant import AsyncQdrantDep


async def endpoint(session: AsyncSessionDep, qdrant: AsyncQdrantDep):
    return await service_function(session, qdrant)
```

Service functions accept ordinary `AsyncSession` and `AsyncQdrantClient` arguments.
They do not need `Request`, `app.state`, or `Depends`.

Each request receives its own PostgreSQL session from the shared
`AsyncSessionLocal` factory. Successful writes require an explicit
`await session.commit()`; errors trigger rollback, and the session closes after
the request. Background jobs can use `async with AsyncSessionLocal() as session`
and manage their transactions explicitly.

Qdrant uses one shared async client and connection pool per worker, created
when its module is imported, like the PostgreSQL engine. Application shutdown
closes Qdrant and disposes the PostgreSQL
engine, including when cleanup of the other resource fails. This setup assumes
one active application lifespan per worker; overlapping app instances in one
process need separate resource ownership. Once closed, the Qdrant client cannot
be reused by another application lifespan; restart the process to create a new client.

PostgreSQL pool settings use `POSTGRES_POOL_SIZE` (default `5`),
`POSTGRES_MAX_OVERFLOW` (default `0`), and `POSTGRES_POOL_TIMEOUT` (default `30`
seconds). Limits apply per worker. Connections are checked before reuse.

Qdrant HTTP pool settings use `QDRANT_MAX_CONNECTIONS` (default `100`),
`QDRANT_MAX_KEEPALIVE_CONNECTIONS` (default `20` idle connections), and
`QDRANT_KEEPALIVE_EXPIRY_SECONDS` (default `5` seconds of idle time).
These limits apply per worker and are passed to the shared client's HTTPX pool.

Models may inherit from `BaseDBModel` to include `created_at` and `updated_at`.
The `updated_at` expression applies to SQLAlchemy-generated updates; it is not a
database trigger for writes from other clients.

## Alembic migrations

Configure `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`,
and `POSTGRES_DATABASE` in `.env`. For host-side commands use
`POSTGRES_HOST=127.0.0.1`; Docker Compose overrides the API's database hostname
to `postgres` and its port to `5432`. Start PostgreSQL and apply the initial
PDF metadata migration:

```bash
docker compose up -d postgres
make migrate
# Equivalent: uv run --locked alembic upgrade head
```

Alembic reads the same database settings as the API. Credentials are not
stored in `alembic.ini`. Online migrations connect asynchronously with
`asyncpg` and bridge Alembic's synchronous operations using `run_sync`.
Existing synchronous connections can also be passed through
`Config.attributes["connection"]` when invoking Alembic programmatically.

After changing ORM models, generate and review a migration, then apply it:

```bash
make migration ARGS='-m "describe schema change"'
make migrate
make migration-check
```

`migration-check` detects differences between the database and registered ORM
models. New model modules must be imported in `alembic/env.py` so their tables
are included. To generate SQL without connecting to PostgreSQL:

```bash
uv run --locked alembic upgrade head --sql
```

For the containerized API, run `docker compose exec api alembic upgrade head`.
Migrations are explicit and do not run automatically at application startup.
Integration tests exercise upgrade, downgrade, and schema comparison on a
disposable PostgreSQL container.

## Commands

```bash
make help                      # List commands
make hooks                     # Enable Git hooks from .githooks for this clone
make build                     # Build the API image
make up                        # Start API (:8000) and Qdrant
make down                      # Stop containers and keep stored data
make logs                      # Follow container logs
make ps                        # Show container status
make config                    # Validate Docker Compose configuration
make lint                      # Run lint checks
make format-check              # Check formatting
make format                    # Format Python files
make test                      # Run unit tests with mocked database calls
make integration               # Run real PostgreSQL tests on host (uv + Docker)
make check                     # Run lint, formatting, and unit tests
make shell                     # Open a shell in the API container
```
