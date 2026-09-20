#!/usr/bin/env bash
# Pull the EC FAQ dataset from the private repo into ./data.
# Uses $GITHUB_TOKEN if set, otherwise falls back to an authenticated `gh` CLI.
set -euo pipefail
REPO="Synesis-IT-PLC/ec-faq-bot"; REF="development"
mkdir -p data
for f in question_tag.csv tag_answer.json; do
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github.raw" \
      "https://api.github.com/repos/$REPO/contents/full_dataset/$f?ref=$REF" -o "data/$f"
  else
    gh api -H "Accept: application/vnd.github.raw" \
      "repos/$REPO/contents/full_dataset/$f?ref=$REF" > "data/$f"
  fi
  echo "fetched data/$f ($(wc -c < "data/$f") bytes)"
done
