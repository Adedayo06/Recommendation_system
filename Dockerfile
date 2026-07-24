# syntax=docker/dockerfile:1
#
# Container image for the recommendation API.
#
# Build (from the project root):
#     docker build -f DockerFile -t recommendation-api .
# Run:
#     docker run --rm -p 8000:8000 recommendation-api
# Then: http://localhost:8000/docs
#
# NOTE: Docker looks for a file literally named "Dockerfile" by default. This
# file is "DockerFile", so either pass -f DockerFile as above, or rename it to
# "Dockerfile" to drop the flag.

FROM python:3.12-slim AS base

# Fail fast, no .pyc files, no pip cache bloating the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first, in their own layer, so a code change doesn't
# re-run pip. Only requirements.txt busts this cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the service needs to SERVE:
#   src/     the model code (persistence, recommender, item_cf, ...)
#   api/     the FastAPI layer
#   models/  the trained artifact (recommender.pkl, ~50 MB)
# The raw data/ CSVs (~1.4 GB) and training scripts are deliberately excluded
# via .dockerignore — load_model() reads only the pickle at runtime.
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Drop root.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The model takes a few seconds to load on startup (the lifespan handler loads
# it before the server accepts traffic), so give the container a grace period
# before health checks count against it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

# One uvicorn worker: each worker loads its own ~50 MB copy of the model, so
# scale out with more containers rather than more workers here. Bump --workers
# only once you've sized the box's memory.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
