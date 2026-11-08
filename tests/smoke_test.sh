#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-/tmp/chimeric_smoke.csv}"
python predict.py --input examples/test_spectra.jsonl --output "$OUT" \
    --checkpoint checkpoints/model.pt --batch-size 1 \
    --device "${DEVICE:-cuda}"
python - "$OUT" <<'PY'
import csv
import sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert len(rows) == 3, rows
assert all(row["peptides"] for row in rows), rows
print("SMOKE_OK", len(rows))
PY
