#!/usr/bin/env bash
set -euo pipefail
find . -type f -size +50M -print0 | xargs -0 -r du -h
