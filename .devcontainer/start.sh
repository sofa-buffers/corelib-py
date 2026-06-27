#!/usr/bin/env bash
set -eo pipefail

# Resolve this script's directory so it works regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
CLAUDE_CONFIG_DIR="$SCRIPT_DIR/.claude-config"
mkdir -p "$CLAUDE_CONFIG_DIR"

env_args=()
if [[ -f "$ENV_FILE" ]]; then
  env_args=(--env-file "$ENV_FILE")
else
  echo "warning: $ENV_FILE not found — copy .devcontainer/.env.example to .devcontainer/.env to load secrets." >&2
fi

docker run -it --rm --name sofa-py-dev \
  "${env_args[@]}" \
  -e CLAUDE_CONFIG_DIR=/root/.claude \
  -v "$(pwd)":/workspace \
  -v "$CLAUDE_CONFIG_DIR":/root/.claude \
  py-devcontainer
