#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
raw="$root/data/raw"
mkdir -p "$raw"

repos=(
  "https://github.com/0xveya/42-codexion|master|42-codexion"
  "https://github.com/0xveya/42-fly-in|master|42-fly-in"
  "https://github.com/sarowish/bawa|master|bawa"
  "https://github.com/etcd-io/etcd|main|etcd"
  "https://github.com/0xveya/gns3util|feature/custom-scripts|gns3util"
  "https://github.com/0xveya/gns3util|feature/orchestration|gns3util2"
  "https://github.com/0xveya/go-to-ts|master|go-to-ts"
  "https://github.com/scolopendraa/gubtool|main|gubtool"
  "https://github.com/k3s-io/k3s|main|k3s"
  "https://github.com/sarowish/mpd-herald|main|mpd-herald"
  "https://github.com/postgres/postgres|master|postgres"
  "https://github.com/tethux/tethux|master|tethux"
  "https://github.com/tokio-rs/tokio|master|tokio"
  "https://github.com/tursodatabase/turso|main|turso"
  "https://github.com/vllm-project/vllm|v0.10.1|vllm-0.10.1"
  "https://github.com/sarowish/ytsub|master|ytsub"
)

for entry in "${repos[@]}"; do
  IFS='|' read -r url branch name <<< "$entry"
  if [[ -e "$raw/$name" ]]; then
    printf 'skipping existing %s\n' "$name" >&2
    continue
  fi
  git clone --branch "$branch" --single-branch "$url" "$raw/$name"
done
