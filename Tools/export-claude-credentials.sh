#!/usr/bin/env bash
# Export the macOS-keychain-stored Claude CLI OAuth token into a file the
# Docker container can mount at /root/.claude/.credentials.json.
#
# The host's ``claude`` CLI stores its OAuth token in the macOS login
# keychain under the service name ``Claude Code-credentials``. A Linux
# container can't read the keychain, so we project the token into a
# file under ``./_docker_claude/`` (gitignored) that docker-compose
# bind-mounts at ``/root/.claude``.
#
# Re-run this whenever you re-login on the host (``claude auth login``)
# -- the token rotates and the container's copy needs to refresh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/_docker_claude"
DEST_FILE="${DEST_DIR}/.credentials.json"

if ! command -v security >/dev/null 2>&1; then
  echo "[export-claude-credentials] this script is macOS-only (uses the security CLI)" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

token_json="$(security find-generic-password -s 'Claude Code-credentials' -w 2>/dev/null || true)"

if [[ -z "${token_json}" ]]; then
  echo "[export-claude-credentials] no Claude Code token found in the keychain." >&2
  echo "[export-claude-credentials] run 'claude auth login' on the host first." >&2
  exit 1
fi

printf '%s' "${token_json}" > "${DEST_FILE}"
chmod 600 "${DEST_FILE}"

# Carry over a minimal settings.json so the container CLI doesn't try to
# bootstrap an empty profile mid-run.
if [[ -f "${HOME}/.claude/settings.json" && ! -f "${DEST_DIR}/settings.json" ]]; then
  cp "${HOME}/.claude/settings.json" "${DEST_DIR}/settings.json"
fi

echo "[export-claude-credentials] wrote ${DEST_FILE}"
echo "[export-claude-credentials] now run: docker compose up -d --build"
