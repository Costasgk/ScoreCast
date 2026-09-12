# ScoreCast — serving image.
#
# This container reads data; it does not fetch it. Scraping needs Chrome,
# undetected-chromedriver and soccerdata, and all of that runs on whatever
# machine produces the CSVs. Keeping it out means the image has no browser, is a
# fraction of the size, and cannot break when a source changes its markup.
#
# Data is mounted at /data rather than copied in, so refreshing predictions never
# requires a rebuild.

FROM python:3.12-slim AS base

# Python in containers: no .pyc clutter, unbuffered logs so `docker logs` is live
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is cached unless requirements change, so code
# edits rebuild in seconds rather than reinstalling pandas and scipy each time.
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Only what serving needs. The registry is shared with the data scripts, and the
# web app imports it from the parent directory.
COPY Scripts/leagues.py       /app/Scripts/leagues.py
COPY Scripts/WebApp           /app/Scripts/WebApp

# The app resolves Datasets relative to its own parent; point that at the mount.
RUN mkdir -p /data /state && ln -s /data /app/Datasets
ENV SCORECAST_STATE_DIR=/state

# Run unprivileged. /data is read-only in normal use, but the visit counter
# writes a SQLite file, so its directory has to belong to this user.
RUN useradd --create-home --uid 10001 scorecast \
 && chown -R scorecast:scorecast /app /data /state
USER scorecast

EXPOSE 8000

# A container that answers is not necessarily a container that works: hit a real
# page so a missing or empty data mount is reported as unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).status == 200 else 1)"

# Two workers is enough: every request reads cached CSVs and returns. The long
# timeout is for /simulator, which fits a model on request.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
     "--workers", "2", "--threads", "4", "--timeout", "120", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "Scripts.WebApp.app:app"]
