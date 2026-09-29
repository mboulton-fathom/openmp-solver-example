# Thin wrapper around uv. The build itself is in CMakeLists.txt, see ./README.md.
OFFLOAD ?=
GPU_ARCH ?= gfx90a

.PHONY: all build test run docker clean

all: build

build:
	uv sync --reinstall-package jacobi-omp \
		-C cmake.define.JACOBI_OFFLOAD=$(OFFLOAD) \
		-C cmake.define.JACOBI_GPU_ARCH=$(GPU_ARCH)

test:
	uv run pytest

run:
	uv run jacobi-solve --grid-size 64

docker:
	docker build -t jacobi-omp .

clean:
	rm -rf build
