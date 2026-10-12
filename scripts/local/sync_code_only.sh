#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOCAL_ROOT="${SPECTRUMCLAW_SYNC_LOCAL_ROOT:-$PROJECT_ROOT}"
REMOTE_HOST="${SPECTRUMCLAW_SYNC_REMOTE_HOST:-weiyifan3090}"
REMOTE_ROOT="${SPECTRUMCLAW_SYNC_REMOTE_ROOT:-/workspace/YiFan/SpectrumClaw}"
FILTER_FILE="$SCRIPT_DIR/sync_code_only.filter"

usage() {
  cat <<'EOF'
用法:
  bash scripts/local/sync_code_only.sh check pull
  bash scripts/local/sync_code_only.sh check push
  bash scripts/local/sync_code_only.sh apply pull
  bash scripts/local/sync_code_only.sh apply push

check 只显示代码差异；apply 才会复制。脚本不使用 --delete。
pull 应用前要求本地 Git 工作区干净；push 应用前要求远端 Git 工作区干净。
EOF
}

MODE="${1:-}"
DIRECTION="${2:-}"
if [[ "$MODE" != "check" && "$MODE" != "apply" ]]; then
  usage >&2
  exit 2
fi
if [[ "$DIRECTION" != "pull" && "$DIRECTION" != "push" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$FILTER_FILE" ]]; then
  echo "同步过滤规则不存在：$FILTER_FILE" >&2
  exit 1
fi
if [[ ! -d "$LOCAL_ROOT" ]]; then
  echo "本地项目目录不存在：$LOCAL_ROOT" >&2
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "缺少 rsync，无法执行代码同步" >&2
  exit 1
fi

if [[ "$MODE" == "apply" && "$DIRECTION" == "pull" ]]; then
  if [[ ! -d "$LOCAL_ROOT/.git" ]]; then
    echo "本地目标不是 Git 工作区：$LOCAL_ROOT" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$LOCAL_ROOT" status --porcelain)" ]]; then
    echo "本地工作区存在未提交改动；为避免覆盖，拒绝 pull。请先提交或另行保存改动。" >&2
    exit 1
  fi
fi

if [[ "$MODE" == "apply" && "$DIRECTION" == "push" ]]; then
  remote_status="$(
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
      "test -d '$REMOTE_ROOT/.git' && git -C '$REMOTE_ROOT' status --porcelain"
  )" || {
    echo "无法确认远端 Git 工作区状态；拒绝 push。" >&2
    exit 1
  }
  if [[ -n "$remote_status" ]]; then
    echo "远端工作区存在未提交改动；为避免覆盖，拒绝 push。" >&2
    exit 1
  fi
fi

rsync_args=(
  --archive
  --compress
  --checksum
  --human-readable
  --itemize-changes
  --safe-links
  --filter="merge $FILTER_FILE"
  -e "ssh -o BatchMode=yes -o ConnectTimeout=15"
)
if [[ "$MODE" == "check" ]]; then
  rsync_args+=(--dry-run)
fi

if [[ "$DIRECTION" == "pull" ]]; then
  source_path="$REMOTE_HOST:$REMOTE_ROOT/"
  target_path="$LOCAL_ROOT/"
else
  source_path="$LOCAL_ROOT/"
  target_path="$REMOTE_HOST:$REMOTE_ROOT/"
fi

echo "模式：$MODE；方向：$DIRECTION"
echo "范围：源码、脚本、配置、测试与文档；排除数据、日志、模型、缓存和构建产物"
rsync "${rsync_args[@]}" "$source_path" "$target_path"
