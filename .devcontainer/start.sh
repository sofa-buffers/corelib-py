#!/usr/bin/env bash
set -eo pipefail

# Resolve this script's directory so it works regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  docker run -it --rm --name sofa-py-dev \
    --env-file "$ENV_FILE" \
    -v "$(pwd)":/workspace \
    -v claude-config:/root/.claude \
    py-devcontainer
else
  echo "warning: $ENV_FILE not found — starting without --env-file." >&2
  echo "         copy .devcontainer/.env.example to .devcontainer/.env to load secrets." >&2
  docker run -it --rm --name sofa-py-dev \
    -v "$(pwd)":/workspace \
    -v claude-config:/root/.claude \
    py-devcontainer
fi
