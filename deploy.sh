#!/bin/bash
set -e

MIN_MEM_KB=$((4 * 1024 * 1024)) # 4 GB

check_memory() {
  if [ -f /proc/meminfo ]; then
    avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ -n "$avail_kb" ]; then
      avail_gb=$(awk -v kb="$avail_kb" 'BEGIN { printf "%.2f", kb/1024/1024 }')
      echo "Available memory: ${avail_gb} GB"
      if [ "$avail_kb" -lt "$MIN_MEM_KB" ]; then
        echo "⚠️  Low available memory (<4GB)."
        echo "    Recommended for WSL stability:"
        echo "    - Close other heavy apps/containers"
        echo "    - Use LOW_MEMORY_MODE=1 and LOW_MEMORY_MAX_CAMERAS=1"
      fi
    fi
  fi
}

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "❌ Docker Compose not found. Install Docker Compose plugin (v2) or docker-compose." >&2
  exit 1
fi

echo "Using compose command: ${COMPOSE_CMD}"
check_memory
${COMPOSE_CMD} build
${COMPOSE_CMD} up -d
${COMPOSE_CMD} logs -f
