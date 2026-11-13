#!/usr/bin/env bash
# Reproduce data/ and vendor/ (both gitignored). Run from repo root.
set -euo pipefail
mkdir -p vendor data
clone() { [ -d "vendor/$1" ] || git clone -q https://github.com/benedikt-schesch/$1 "vendor/$1"; git -C "vendor/$1" checkout -q "$2"; }
clone Merge-Bench         $(grep '^Merge-Bench ' scripts/PINS | cut -d' ' -f2)
clone Merge-Bench-Builder $(grep '^Merge-Bench-Builder ' scripts/PINS | cut -d' ' -f2)
clone LLMerge             $(grep '^LLMerge ' scripts/PINS | cut -d' ' -f2)
cd data
[ -f merges.tar.gz ] || curl -sL -o merges.tar.gz https://github.com/benedikt-schesch/Merge-Bench/releases/download/main/merges.tar.gz
echo "4f0b63409776c7d987b808dd85959c40a09160c6f155381b66a9ed8356ea6ff8  merges.tar.gz" | sha256sum -c -
mkdir -p merges && tar xzf merges.tar.gz -C merges
