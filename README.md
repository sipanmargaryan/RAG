# RAG

This FastAPI application processes PDF documents and finds passages relevant to a
natural-language query. It uses query expansion to broaden retrieval and
CrossEncoder reranking to rank the retrieved passages by relevance. It provides
document ingestion, retrieval, and answer generation for a retrieval-augmented
generation (RAG) system, returning an answer with citations and source document
metadata.


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
