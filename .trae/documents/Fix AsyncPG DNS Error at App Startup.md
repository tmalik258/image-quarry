## Root Cause
- The FastAPI app initializes the DB at startup (`app/main.py:30-37`), using an async SQLAlchemy engine created from `settings.DATABASE_URL` (`app/database.py:12-19`).
- In Docker, `DATABASE_URL` is set to `postgresql+asyncpg://postgres:password@db:5432/image_quarry` (`docker-compose.yml:10,39,64`).
- The error `socket.gaierror: [Errno -2] Name or service not known` indicates the hostname in `DATABASE_URL` could not be resolved when asyncpg tried to connect.
- Most likely causes:
  1) App container is not seeing the compose-provided `DATABASE_URL` and is falling back to a wrong/default value (see default in `app/config.py:23-26`).
  2) Compose/network mismatch where service hostname `db` is temporarily not resolvable for the `app` container.
  3) Startup race: app attempts connection before Docker network/DB service is ready.

## Verification Steps
1. Confirm the active `DATABASE_URL` inside the running app container:
   - `docker compose exec app /bin/sh -c "echo $DATABASE_URL"`
2. Verify the app and db share the same network:
   - `docker compose ps` then `docker inspect <app-container-id> | jq '.[0].NetworkSettings.Networks'`
3. Check db health and logs:
   - `docker compose logs -f --tail=50 db`
4. If `DATABASE_URL` differs from compose (e.g., shows `localhost`), it means `.env` or an override is winning over compose envs.

## Fix Plan
- Ensure the container uses the compose-provided URL:
  - If step (1) shows the wrong host, update compose to set `env_file: .env` only for local non-Docker runs or remove/rename `DATABASE_URL` in `.env` for Docker usage. Environment in compose should override `.env` but we will verify precedence.
- Harden startup ordering:
  - Add `depends_on` with health conditions so `app` waits until `db` is healthy:
    ```yaml
    app:
      depends_on:
        db:
          condition: service_healthy
    ```
  - `db` already has a healthcheck (`docker-compose.yml:86-90`), so the condition will gate app startup.
- Add connection retry logic around `init_db()`:
  - Implement exponential backoff (e.g., 10 attempts, 3s–30s) specifically catching `socket.gaierror` and asyncpg connection errors before failing the app startup.
- Validate end-to-end:
  - `docker compose up -d`, confirm `app` health (`docker-compose.yml:28-33`), and hit `http://localhost:8000/health/`.

## Deliverables (once approved)
- Compose update with `depends_on: condition: service_healthy` for `app/worker/scheduler`.
- Robust retry wrapper for `init_db()` in `app/main.py` or `app/database.py` without exceeding 200 lines per file.
- Short README note on env precedence: compose env vs `.env` for local dev.

Proceed to apply the verification and fixes?