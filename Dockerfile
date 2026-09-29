FROM python:3.13-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends clang libomp-dev cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV CXX=clang++
WORKDIR /src
COPY . .
RUN uv build --wheel --python /usr/local/bin/python3 --out-dir /dist
RUN uv export --locked --no-dev --no-emit-project --output-file /dist/requirements.txt


FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libomp5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /dist /tmp/dist
RUN pip install --no-cache-dir -r /tmp/dist/requirements.txt \
    && pip install --no-cache-dir --no-deps /tmp/dist/*.whl \
    && rm -rf /tmp/dist

WORKDIR /work
ENTRYPOINT ["jacobi-solve"]
