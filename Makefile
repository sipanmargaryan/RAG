.DEFAULT_GOAL := help

COMPOSE ?= docker compose
ARGS ?=
UV ?= uv

.PHONY: help hooks build up down logs ps config lint format-check format test integration check shell migrate migration migration-check qdrant-migrate qdrant-migration-check

help:
	@printf '%s\n' \
	  'Commands:' \
	  'make help                     List all Make commands' \
	  'make build                    Build the API image' \
	  'make up                       Start all services' \
	  'make down                     Stop containers and keep stored data' \
	  'make logs                     Follow container logs' \
	  'make ps                       Show container status' \
	  'make config                   Validate Docker Compose configuration' \
	  'make lint                     Run Ruff lint checks' \
	  'make format-check             Check Python formatting' \
	  'make format                   Format source files on the host using the container' \
	  'make test                     Run unit tests without a database; accepts ARGS="..."' \
	  'make integration              Run Testcontainers tests on the host (requires uv and Docker)' \
	  'make check                    Validate Compose and run tox; accepts ARGS="..."' \
	  'make shell                    Open a shell in the API container' \
	  '' \
	  'Git hooks:' \
	  'make hooks                    Enable Git hooks from .githooks for this clone' \
	  '' \
	  'Database migrations:' \
	  'make migration                Generate a migration; ARGS="-m description"' \
	  'make migrate                  Apply Alembic migrations to PostgreSQL' \
	  'make migration-check          Check database schema against ORM models' \
	  '' \
	  'Qdrant migrations:' \
	  'make qdrant-migrate           Create or migrate the Qdrant collection' \
	  'make qdrant-migration-check   Validate the Qdrant collection schema'

hooks:
	git config --local core.hooksPath .githooks

build:
	$(COMPOSE) build api

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

config:
	$(COMPOSE) config --quiet

lint: build
	$(COMPOSE) run --rm api ruff check .

format-check: build
	$(COMPOSE) run --rm api ruff format --check .

format: build
	$(COMPOSE) run --rm --user "$$(id -u):$$(id -g)" \
	  --volume "$(CURDIR):/app" api ruff format --no-cache .

test: build
	$(COMPOSE) run --rm --no-deps api pytest -n auto $(ARGS)

integration:
	$(UV) run --locked pytest -m integration -n auto $(ARGS)

migrate:
	$(UV) run --locked alembic upgrade head

migration:
	$(UV) run --locked alembic revision --autogenerate $(ARGS)

migration-check:
	$(UV) run --locked alembic check

qdrant-migrate:
	$(COMPOSE) exec -T api python -m app.migrations.qdrant

qdrant-migration-check:
	$(COMPOSE) exec -T api python -m app.migrations.qdrant --check

check: config build
	$(COMPOSE) run --rm --no-deps api tox $(ARGS)

shell: build
	$(COMPOSE) run --rm api sh
