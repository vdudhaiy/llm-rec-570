#!/bin/bash
# Same LLMRec architecture and hyperparameters throughout; only the text changes.
# This is the comparison the write-up's Table 1 is meant to report.
# Result filenames are bare: config.output_path() routes them into OUTPUT_DIR
# (default outputs/), so nothing here needs to know the directory layout.
set -u
PY=.venv/Scripts/python.exe
export VERBOSE=True

echo "=== baseline: original TMDB descriptions ==="
EMBEDDINGS_FILE_OVERRIDE=movie_embeddings.json $PY -u training.py -m B -e 50 \
  --results-file results_cmp_original.json 2>&1 | grep -E "Test Results|HR@10:|MRR@10:|NDCG@10:|Test Loss|Early stopping"

echo "=== C: recommendation-driven prompting ==="
$PY -u training.py -m C -e 50 --results-file results_cmp_C.json 2>&1 \
  | grep -E "Test Results|HR@10:|MRR@10:|NDCG@10:|Test Loss|Early stopping"

echo "=== D: combined (concatenated, encoder truncates) ==="
$PY -u training.py -m D -e 50 --results-file results_cmp_D.json 2>&1 \
  | grep -E "Test Results|HR@10:|MRR@10:|NDCG@10:|Test Loss|Early stopping"

echo "=== E: multi-view attention ==="
$PY -u training.py -m E -e 50 --results-file results_cmp_E.json 2>&1 \
  | grep -E "Test Results|HR@10:|MRR@10:|NDCG@10:|Test Loss|Early stopping|Mean attention|^ +(original|basic|rec_driven)"

echo "=== A: SimpleCF, no text ==="
$PY -u training.py -m A -e 50 --results-file results_cmp_A.json 2>&1 \
  | grep -E "Test Results|HR@10:|MRR@10:|NDCG@10:|Test Loss|Early stopping"

echo "COMPARISON DONE"
