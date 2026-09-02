# LLM-Rec Reimplementation

A reimplementation of **LLM-Rec: Personalized Recommendation via Prompting Large Language Models**
(Lyu et al., *Findings of NAACL 2024*) on the MovieLens-1M dataset.

The idea under test: if you rewrite each item's text description with an LLM before
embedding it, does a content-based recommender get better? The pipeline fetches real
synopses from TMDB, rewrites them with an LLM through Purdue GenAI Studio, embeds them
with a sentence transformer, and trains a small attention-based ranker on the result.

## Project Structure

```
llm-rec-570/
├── main.py                  # Pipeline orchestrator (MAIN ENTRY POINT)
├── model.py                 # Neural network models (SimpleCF, LLMRec) + EarlyStopping
├── metrics.py               # Evaluation metrics (HR, MRR, NDCG)
├── training.py              # Model training, evaluation, result serialization
├── data_processing.py       # Dataset creation and embedding generation
├── fetching_descriptions.py # Fetch movie descriptions from TMDB API
├── prompting.py             # LLM-based description enhancement
├── config.py                # Centralized configuration (loads from .env)
├── checkpoint.py            # Resume support: what is done, what is still owed
├── test_resume.py           # Unit tests for the resume logic (no network)
├── test_pipeline_skip.py    # End-to-end skip/resume tests (network stubbed)
├── outputs/                 # All run results land here automatically
│   ├── *.json               #   test metrics + per-epoch history
│   ├── logs/                #   run logs (gitignored)
│   └── archive/             #   superseded results from earlier runs
├── run_comparison.sh        # Reproduces the results table below
├── requirements.txt         # Python dependencies (torch installed separately)
├── .env                     # Credentials — gitignored, create your own
└── README.md                # This file
```

Data and generated artefacts are **not** in the repo (see `.gitignore`); you produce
them by running the pipeline:

| Artefact | Produced by | Contents |
|---|---|---|
| `movielens-1m/` | manual download | `ratings.dat`, `movies.dat`, `users.dat` |
| `movies_desc.dat` | Step 1 | `movieId::title::year::genres::TMDB synopsis` |
| `basic_descriptions.txt` | Step 2 | LLM summaries, one per movie, `\n::\n`-separated |
| `recommendation_driven_descriptions.txt` | Step 2 | LLM recommendation pitches, same format |
| `movies_enhanced_desc_combined.dat` | Step 2 | Merged descriptions in `.dat` format |
| `movie_embeddings.json` | Step 3 | `{movieId: [384 floats]}` from original descriptions |
| `embeddings_basic.json` | Step 3 | …from basic descriptions |
| `embeddings_rec_driven.json` | Step 3 | …from recommendation-driven descriptions |
| `embeddings_combined.json` | Step 3 | …from combined descriptions |
| `embeddings_multiview.json` | Step 3 | one embedding **per view**, stacked: `{"views": [...], "embeddings": {movieId: [[dim], [dim], ...]}}` |
| `outputs/test_results_<TYPE>.json` | Step 5 | Test metrics + per-epoch history |

## Model Types

| Type | Model | Embeddings used | Role |
|------|-------|-----------------|------|
| **A** | `SimpleCF` | none (learns its own) | Pure collaborative-filtering reference point |
| **B** | `LLMRec` | `embeddings_basic.json` | Basic (summarization) prompting |
| **C** | `LLMRec` | `embeddings_rec_driven.json` | Recommendation-driven prompting |
| **D** | `LLMRec` | `embeddings_combined.json` | Combined descriptions (concatenated into one string) |
| **E** | `LLMRec` | `embeddings_multiview.json` | Each description embedded separately; attention picks between them |

> **Note on the baseline.** Type A is a *different architecture* (matrix factorization),
> not the paper's baseline. The write-up compares a baseline trained on the **original
> TMDB descriptions** against LLM-Rec trained on the **enhanced** ones — same model, different
> text. To reproduce that comparison, run `LLMRec` twice with `EMBEDDINGS_FILE_OVERRIDE`,
> which swaps the text without swapping the architecture:
>
> ```bash
> EMBEDDINGS_FILE_OVERRIDE=movie_embeddings.json python training.py -m B -e 5 --results-file results_original.json
> ```
> ```bash
> EMBEDDINGS_FILE_OVERRIDE=embeddings_basic.json python training.py -m B -e 5 --results-file results_enhanced.json
> ```
>
> Type A answers a separate question: how much does the text buy you over no text at all?


### Type E: multi-view attention

Types B/C/D give each movie **one** vector. The attention layer therefore attends over a
sequence of length 1, where softmax is always exactly 1.0 — it cannot make a choice, and
degenerates into a learned linear projection of the movie embedding. The user embedding
has no influence on its output.

Type E keeps each description as its **own** vector:

```
movie = [ encode(synopsis), encode(basic summary), encode(rec pitch) ]
```

Now the user embedding is a real query. Attention scores the user against each view,
softmaxes into percentages, and blends them — and the percentages differ from user to
user. The model learns *per user* which kind of description predicts their taste.

Three things fall out of this:

1. **`mean_attention_weights` is a reportable result.** Instead of inferring which
   prompting strategy helped by comparing final scores across separate training runs,
   the model states directly how much weight it placed on each description type. This is
   printed after the test evaluation and stored in `outputs/test_results_E.json`.
2. **No extra LLM calls.** Step 3 stacks the per-view embedding files it already built.
   `generateCombinedDescriptions` — a third pass over every movie, ~5 hours of API time —
   is not needed for Type E.
3. **No truncation.** `paraphrase-MiniLM-L6-v2` truncates at **128 tokens**. Measured on
   the generated combined descriptions, the concatenated text averages **127 tokens and
   reaches 225**, so **1718 of 3883 movies (44.2%)** are silently cut off mid-sentence.
   Embedding views separately gives each its own full budget. This is not theoretical:
   Type D is the only variant that scores *below* the no-LLM baseline (see Results).

Build it and train it:

```bash
python main.py --step 3
```
```bash
python training.py -m E -e 10 --results-file test_results_E.json
```

Which views are stacked is controlled by `MULTIVIEW_SOURCES` in `config.py`. Views whose
embeddings file is absent are skipped with a warning, and at least two are required.

> **Interpreting the weights.** If a view's text is missing, Step 3 falls back to the
> original TMDB description for that movie — so the two views encode the *same string*
> and attention splits 50/50 by symmetry. A reported 50/50 means "the views are
> identical", not "both descriptions mattered equally". Check the coverage percentage
> Step 3 logs before reading anything into the weights.

## Installation

### Prerequisites
- Python 3.10+
- An NVIDIA GPU is strongly recommended (CPU works but is roughly 20× slower)
- TMDB API key (Step 1 only) and Purdue GenAI Studio access (Step 2 only)

### Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1     # Windows PowerShell
   source .venv/bin/activate      # Linux/macOS
   ```

2. Install PyTorch for your GPU **first**, then the rest. Pick the CUDA build that
   matches your card — Blackwell cards (RTX 50-series) need cu128 or newer:
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu128 torch
   ```
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the GPU is visible:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

4. Create `.env`:
   ```env
   # TMDB (Step 1 only)
   TMDB_API_KEY=<your_tmdb_api_key>

   # Purdue GenAI Studio (Step 2 only)
   PURDUE_API_URL=https://genai.rcac.purdue.edu/api/chat/completions
   PURDUE_GEN_AI_KEY=<your_purdue_genai_key>
   MODEL=gemma3:27b

   # Model selection
   MODEL_TYPE=B

   # Training
   TRAIN_EPOCHS=50
   LEARNING_RATE=1e-4
   BATCH_SIZE=64
   RANDOM_SEED=42
   VERBOSE=True
   ```
   Missing TMDB/Purdue credentials only disable Steps 1 and 2 — training still runs
   as long as the description files already exist.

5. Download MovieLens-1M from <https://grouplens.org/datasets/movielens/1m/> and
   extract it to `./movielens-1m/`.

## Usage

```bash
python main.py                    # full pipeline, reusing finished work
```

### Resuming and redoing work

Steps 1 and 2 make one network call per movie — about half an hour against TMDB
and over ten hours against the LLM endpoint. Neither ever repeats work it has
already paid for:

* **Completed work is skipped.** A finished pass costs zero API calls on a rerun.
* **Partial work is resumed.** A file with 441 of 3883 entries is recognised as
  incomplete, and only the remaining 3442 are requested.
* **Interrupted runs keep their progress.** Results are checkpointed to disk every
  10 items and again on the way out, including on Ctrl-C. Writes are atomic, so a
  crash mid-checkpoint cannot corrupt what was already banked.

```bash
python main.py --force            # ignore everything on disk and regenerate
```
```bash
python main.py --retry-failed     # also re-request entries that previously failed
```

`--force` and `--retry-failed` are different: `--force` redoes *everything*, while
`--retry-failed` redoes only entries recorded as `[ERROR]` or `NOT_FOUND`. Failure
retry is off by default, because some movies genuinely have no upstream data and
would be re-requested on every single run.

### Rate limiting the LLM endpoint

Purdue GenAI Studio throttles with **HTTP 400** and a body of
`{"detail": "Rate limit exceeded. Please try again later."}` — *not* the 429 the
retry path originally looked for. A 400 that is really a throttle must not be
treated as a bad request: doing so burns the retry budget and can lose a
description permanently.

`api_call_with_retry` therefore matches on the message as well as the status code,
and throttling gets its own, larger retry budget (`RETRY_RATE_LIMIT_ATTEMPTS`,
default 12) separate from the budget for real failures. `Retry-After` is honoured
when the server sends it.

The sustainable request rate is undocumented and varies with load, so the delay is
**adaptive**. It starts at `PROMPT_RATE_LIMIT`, multiplies by 1.5 each time the
server pushes back (capped at `PROMPT_RATE_LIMIT_MAX`), and eases back down by 10%
after 300 consecutive clean calls.

| Variable | Default | Meaning |
|---|---|---|
| `PROMPT_RATE_LIMIT` | 2.5 | Starting and minimum seconds between LLM calls |
| `PROMPT_RATE_LIMIT_MAX` | 20.0 | Ceiling the adaptive delay will not exceed |

Set `PROMPT_RATE_LIMIT` higher if you see a lot of `Server reported a rate limit`
warnings early in a run — starting above the sustainable rate wastes time in
backoff, and the climb to a workable delay is slower than just beginning there.

The same flags work on the individual scripts:

```bash
python prompting.py --skip-combined      # resume; model type E needs no combined pass
```
```bash
python fetching_descriptions.py --force  # refetch every TMDB synopsis
```

Verify the behaviour without spending an API call:

```bash
python test_resume.py && python test_pipeline_skip.py
```

```bash
python main.py --model-type A     # SimpleCF
python main.py --model-type D     # LLMRec + combined descriptions
python main.py --model-type E     # LLMRec + multi-view attention
```

Run one step at a time:

```bash
python main.py --step 1           # fetch TMDB descriptions (resumes)
```
```bash
python main.py --step 2           # LLM description enhancement (slow, API-bound, resumes)
```
```bash
python main.py --step 3           # generate all four embedding files
```
```bash
python main.py --step 4           # build and validate the train/val/test splits
```
```bash
python main.py --step 5 -m D -e 20  # train and test type D for 20 epochs
```

Or drive training directly:

```bash
python training.py --model-type B --epochs 10 --results-file test_results_B.json
```

### Quick trial runs

`SUBSAMPLE_FRAC` keeps a random fraction of the ratings file, which turns a
multi-minute epoch into a couple of seconds:

```bash
SUBSAMPLE_FRAC=0.05 python training.py -m B -e 3
```

## Configuration

All parameters live in `config.py` and are overridable through `.env`.

### Training
| Variable | Default | Meaning |
|---|---|---|
| `TRAIN_EPOCHS` | 50 | Maximum epochs |
| `LEARNING_RATE` | 1e-4 | AdamW learning rate |
| `BATCH_SIZE` | 64 | Positives per batch (each expands to 1 + `NUM_NEGATIVES` scored candidates) |
| `NUM_NEGATIVES` | 9 | Negatives per positive **during training** |
| `EARLY_STOPPING_PATIENCE` | 10 | Epochs without validation-loss improvement before stopping |
| `WEIGHT_DECAY` | 1e-5 | AdamW weight decay |

### Evaluation
| Variable | Default | Meaning |
|---|---|---|
| `EVALUATION_K` | 10 | Cutoff for HR/MRR/NDCG |
| `EVAL_NUM_NEGATIVES` | 99 | Negatives per positive **at evaluation time** |
| `EVAL_MAX_BATCHES` | 0 | Cap evaluation batches (0 = full split) |
| `POSITIVE_RATING_THRESHOLD` | 2.0 | Ratings at or above this count as positive feedback |

> `EVAL_NUM_NEGATIVES` must be greater than `EVALUATION_K - 1`. With 9 negatives and
> K=10 every candidate fits inside the top-10 window, so HR@10 is pinned at 1.0 for
> *every* model regardless of quality. 99 negatives gives a
> random-guess floor of HR@10 = 0.10, which is the standard leave-one-out protocol.

### Device / performance
| Variable | Default | Meaning |
|---|---|---|
| `DEVICE_TYPE` | `auto` | `auto`, `cuda`, or `cpu` |
| `USE_AMP` | `True` | bfloat16 autocast for the training forward pass |
| `USE_TF32` | `True` | TF32 matmuls on Ampere+ GPUs |
| `NUM_WORKERS` | 0 | DataLoader workers (0 is safest on Windows) |
| `PIN_MEMORY` | `True` | Pinned host memory for faster host→device copies |
| `SUBSAMPLE_FRAC` | 1.0 | Fraction of the ratings file to keep |
| `OUTPUT_DIR` | `outputs` | Directory all result files are written to |
| `EMBEDDINGS_FILE_OVERRIDE` | unset | Force a specific embeddings file, independent of `MODEL_TYPE` |
| `MULTIVIEW_SOURCES` | 3 views | *(config.py only)* Which single-view files feed the Type E stack |

## Results

Full run on MovieLens-1M, 2026-09-01. Every row uses the **same `LLMRec` architecture
and the same hyperparameters** — 50-epoch budget, lr 1e-4, batch 64, 9 training
negatives, early-stopping patience 10. The only thing that changes between rows is
the **text** the descriptions were built from. Type A is the exception and is listed
separately, because it is a different architecture (matrix factorization, no text).

Evaluation: each test interaction ranked against **99 sampled negatives** the user
has not rated positively, cutoff K=10. Hardware: RTX 5070 Laptop (sm_120), CUDA 12.8.

| Variant | HR@10 | MRR@10 | NDCG@10 | test loss | epochs | vs baseline |
|---|---|---|---|---|---|---|
| baseline — original TMDB text | 0.5641 | 0.2400 | 0.3158 | 0.0971 | 31 | — |
| **B** — basic prompting | 0.5857 | 0.2529 | 0.3308 | 0.0930 | 50 | **+4.66%** |
| **C** — recommendation-driven | 0.5949 | 0.2566 | 0.3358 | 0.0918 | 44 | **+6.24%** |
| **D** — combined (concatenated) | 0.5351 | 0.2259 | 0.2981 | 0.0979 | 18 | **−5.54%** |
| **E** — multi-view attention | 0.5917 | 0.2585 | 0.3365 | 0.0920 | 42 | **+6.39%** |
| *A* — SimpleCF, no text | *0.5075* | *0.2109* | *0.2800* | *0.1041* | *17* | *−11.16%* |
| random guessing | 0.1000 | 0.0293 | 0.0436 | — | — | — |

Every row satisfies MRR@10 ≤ NDCG@10 ≤ HR@10.

Reproduce with:

```bash
./run_comparison.sh
```

### What the numbers say

**1. LLM-enhanced descriptions do help — by about 5–6%.** Both single-prompt variants
beat the original TMDB synopses on every metric, and test loss improves alongside the
ranking metrics, so this is not an artefact of the ranking measure.

**2. Recommendation-driven prompting beats basic prompting** (+6.24% vs +4.66%). This
reproduces the ordering reported in the original LLM-Rec paper. It is also intuitive:
a recommendation pitch describes *why someone would enjoy this item*, which is closer
to the ranking objective than a plot summary is.

**3. Concatenating descriptions actively hurts (−5.54%) — because of encoder
truncation, not prompting.** `paraphrase-MiniLM-L6-v2` truncates at 128 tokens. The
generated combined text averages **127 tokens and runs to 225**, so **44.2% of movies
(1718/3883) lose text**, cut off mid-sentence. Type D is the only variant that loses
to doing nothing, and it costs an extra 3883 LLM calls (~5.4 hours) to get there. It
also early-stops soonest (18 epochs), consistent with degraded embeddings.

**4. The same three descriptions, packaged differently, swing the result by 12.6
points.** Type D and Type E consume *identical* text. D concatenates and truncates
(−5.54%); E embeds each view separately, giving each its own full 128-token budget,
and lets attention weight them (+6.39%). E also skips the third LLM pass entirely.

**5. Attention says the model relies on recommendation-driven text.** Averaged over
the test set, Type E distributes its attention as:

| view | mean attention weight |
|---|---|
| `rec_driven` | **52.3%** |
| `original` | 24.8% |
| `basic` | 22.9% |

The model puts more weight on the recommendation pitch than on the other two views
combined — an independent confirmation of finding (2), arrived at from a single
training run rather than by comparing separate ones.

**6. Text matters on this dataset, but collaborative filtering is a strong floor.**
SimpleCF, with no text at all, reaches HR@10 0.5075 — well above chance and only 11%
below the text baseline. MovieLens-1M is dense (~165 ratings per user), so the
interaction matrix alone carries most of the signal. LLM-Rec's premise should pay off
more in sparse or cold-start settings; a dense benchmark understates the effect.

### Caveats

- **Single seed, no error bars.** C (+6.24%) and E (+6.39%) differ by 0.15pp, which one
  run cannot separate. The honest reading is "C and E are comparable, and both beat
  basic prompting". Repeat across seeds before treating any gap that small as real.
- **Unequal effective budgets.** Early stopping ended runs at different epochs
  (18–50), so variants did not all train equally long. This is the criterion behaving
  as designed, not a bug, but it is a confound in a strict comparison.
- **Type A is a different architecture**, so its row measures "does text help at all",
  not "does prompting help".

### Relationship to earlier reported figures

An earlier version of this project reported +3.355% from a table whose rows violated
MRR ≤ HR, using metrics with two defects: `mrr_at_k` and `ndcg_at_k` truncated their
sums to integers (collapsing both onto HR@1), and evaluation used 9 negatives at K=10,
which makes HR@10 identically 1.0 for every model. Both are fixed; see
`test_resume.py` and the metric identities noted above. The numbers in this section
supersede that table.

## Evaluation Metrics

Each test interaction is ranked against `EVAL_NUM_NEGATIVES` sampled negatives that the
user has not rated positively. Results land in **`outputs/`** — `config.output_path()` routes every relative
results path there and creates the directory on demand, so nothing needs to be
passed on the command line:

```bash
python training.py -m C -e 50                      # -> outputs/test_results_C.json
```
```bash
python training.py -m C --results-file seed1.json  # -> outputs/seed1.json
```
```bash
python training.py -m C --results-file sweep/s1.json  # -> outputs/sweep/s1.json
```

Absolute paths are honoured as given, and a path that already names `outputs/` is
not nested a second time. Override the directory with `OUTPUT_DIR` in `.env`.

Intermediate data — descriptions and embeddings — deliberately stays at the project
root, because those are *inputs* to later steps and the resume and staleness checks
look for them there.

Example `outputs/test_results_<TYPE>.json`:

```json
{
  "model_type": "B",
  "model_description": "LLMRec with basic descriptions",
  "device": "NVIDIA GeForce RTX 5070 Laptop GPU (sm_120, 12.0 GB, CUDA 12.8)",
  "epochs_run": 5,
  "eval_num_negatives": 99,
  "test_metrics": {
    "HR@10": 0.3162,
    "MRR@10": 0.1291,
    "NDCG@10": 0.1723
  },
  "views": ["original", "basic"],
  "mean_attention_weights": {"original": 0.51, "basic": 0.49},
  "average_test_loss": 0.0921,
  "history": [ { "epoch": 1, "train_loss": 0.31, "val_loss": 0.10, "seconds": 92.4 } ]
}
```

### Definitions
- **HR@k** — fraction of test interactions whose true item lands in the top-k.
- **MRR@k** — mean of 1/rank of the true item when it appears in the top-k.
- **NDCG@k** — position-discounted gain. With one relevant item per query the ideal DCG
  is 1, so DCG and NDCG coincide.
**Precision@k is not reported.** Each query has exactly one relevant item, so
Precision@k = HR@k / k identically — it is HR@k on a different scale, not a separate
result. If you need it to line up with a table that quotes it (the original paper does),
divide HR@k by k.

Sanity check for any reported table: **MRR@k ≤ NDCG@k ≤ HR@k**. A table that violates
this has a metric bug.

## Implementation Notes

- **Implicit feedback.** MovieLens ratings are explicit (1–5). A rating of
  `POSITIVE_RATING_THRESHOLD` or higher becomes a label of 1; observed interactions
  below the threshold are treated as negatives, not positives.
- **Negative sampling.** Negatives are drawn uniformly and rejected if they hit the true
  item or anything in the user's positive set, using a dense boolean
  `(num_users, num_movies)` matrix that lives on the GPU.
- **Attention layer.** For types B/C/D, `LLMRec` attends from the user embedding to a
  *single* movie vector. A softmax over one element is always 1.0, so the layer reduces
  to a learned linear projection and the user has no influence on its output. These types
  are kept to match the originally reported architecture. **Type E** feeds several view
  vectors instead, which makes the attention genuine — see *Type E: multi-view attention*.
- **Movie ID indexing.** MovieLens IDs run 1–3952 and are used directly as embedding
  indices, so `NUM_MOVIES` is 3953 (index 3952 must be valid).
- **Reproducibility.** `RANDOM_SEED` seeds Python, NumPy, and Torch. Validation
  negatives are drawn from a separately seeded generator so the per-epoch numbers are
  comparable to each other.
- **Cost of Step 2.** `prompting.py` rate-limits to one API call every 5 seconds.
  Three passes over 3883 movies is roughly 16 hours; `--skip-combined` drops it to
  about 11. The pass is resumable, so it is safe to stop and restart. Step 3 fills any
  remaining gap with the original TMDB text and logs the coverage percentage.
- **Encoding.** MovieLens-1M ships as ISO-8859-1, not UTF-8 — titles like
  `Misérables, Les (1995)` make a UTF-8 read raise `UnicodeDecodeError`.
- **Security.** `.env` is gitignored. Never commit API keys.

## Reference

Hanjia Lyu, Song Jiang, Hanqing Zeng, Yinglong Xia, Qifan Wang, Si Zhang, Ren Chen,
Chris Leung, Jiajie Tang, and Jiebo Luo. *LLM-Rec: Personalized recommendation via
prompting large language models.* Findings of ACL: NAACL 2024, pages 583–612.
