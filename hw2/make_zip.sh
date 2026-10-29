#!/bin/bash
# Packages the submission: LASTNAME_FIRSTNAME_HW2.zip (code, configs, results,
# videos, report; no data/checkpoints/venv/secrets).
#   bash make_zip.sh SHARMA HARSH
set -euo pipefail
NAME="${1:?LASTNAME}_${2:?FIRSTNAME}_HW2"
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
mkdir "$TMP/$NAME"
cp -r "$HERE"/{README.md,requirements.txt,code,configs,slurm,results,videos} "$TMP/$NAME/"
[ -f "$HERE/report/report.pdf" ] && cp "$HERE/report/report.pdf" "$TMP/$NAME/" || echo "WARNING: report/report.pdf missing"
find "$TMP/$NAME" -name "__pycache__" -prune -exec rm -rf {} +
if grep -rIlE "(api[_-]?key|WANDB_API_KEY|password)\s*[:=]" "$TMP/$NAME"; then
  echo "Possible secret found above; aborting."; exit 1
fi
(cd "$TMP" && zip -qr "$HERE/../$NAME.zip" "$NAME")
rm -rf "$TMP"
unzip -l "$HERE/../$NAME.zip" | tail -1
echo "Wrote $(cd "$HERE/.." && pwd)/$NAME.zip"
