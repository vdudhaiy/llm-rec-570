"""
LLM-based description enhancement via Purdue GenAI Studio.

Every pass here is resumable. One call per movie at a five-second rate limit means
a full pass over MovieLens-1M takes hours, so a run that dies partway must never
throw away what it already paid for. Descriptions are checkpointed to disk as they
arrive, and a rerun regenerates only the movies still missing a usable result.
Pass force=True (or --force) to start over from scratch.
"""

import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer

from checkpoint import (
    load_entries, save_entries, load_dat_rows, save_dat_rows,
    plan_work, is_placeholder, CHECKPOINT_EVERY
)
from config import (
    LLM_MODEL, PURDUE_API_URL, PURDUE_API_KEY,
    TEMPERATURE, MAX_NEW_TOKENS, LLM_BATCH_SIZE,
    MOVIES_DESC_FILE, BASIC_DESC_FILE, REC_DRIVEN_DESC_FILE,
    MOVIES_ENHANCED_DESC_COMBINED, EMBEDDING_MODEL, VERBOSE, RANDOM_SEED
)

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adaptive rate limiting
# ---------------------------------------------------------------------------
# Purdue GenAI Studio signals throttling with HTTP 400 and a body of
# {"detail": "Rate limit exceeded. Please try again later."} - NOT 429. Treating
# it as a generic client error burns the retry budget and can lose a description
# permanently, so the check below matches on the message as well as the status.
#
# The sustainable rate is not documented and appears to vary, so rather than
# hardcoding a guess the delay starts at PROMPT_RATE_LIMIT and climbs whenever the
# server pushes back, easing off again after a long clean streak.
RATE_LIMIT_MIN = float(os.getenv("PROMPT_RATE_LIMIT", "2.5"))       # starting/floor delay
RATE_LIMIT_MAX = float(os.getenv("PROMPT_RATE_LIMIT_MAX", "20.0"))  # ceiling
RATE_LIMIT_GROWTH = 1.5       # multiplier applied when throttled
RATE_LIMIT_DECAY = 0.9        # multiplier applied after a long clean streak
RATE_LIMIT_DECAY_AFTER = 300  # consecutive successes before easing off

RATE_LIMIT_DELAY = RATE_LIMIT_MIN  # current delay; adapts at runtime

RETRY_MAX_ATTEMPTS = 5
RETRY_RATE_LIMIT_ATTEMPTS = 12  # throttling is transient, so be far more patient
RETRY_INITIAL_BACKOFF = 3.0
RETRY_MAX_BACKOFF = 60.0
last_api_call_time = 0.0
_consecutive_ok = 0


def _looks_rate_limited(response):
    """
    True if `response` is the server telling us to slow down.

    Args:
        response: A requests.Response

    Returns:
        bool: True for a 429, or a 4xx whose body mentions a rate limit
    """
    if response.status_code == 429:
        return True
    if response.status_code in (400, 403, 503):
        try:
            return "rate limit" in response.text.lower()
        except Exception:
            return False
    return False


def _note_throttled():
    """Increase the inter-call delay after the server pushed back."""
    global RATE_LIMIT_DELAY, _consecutive_ok
    _consecutive_ok = 0
    previous = RATE_LIMIT_DELAY
    RATE_LIMIT_DELAY = min(RATE_LIMIT_DELAY * RATE_LIMIT_GROWTH, RATE_LIMIT_MAX)
    if RATE_LIMIT_DELAY > previous:
        logger.warning(f"Server reported a rate limit; slowing down "
                       f"{previous:.2f}s -> {RATE_LIMIT_DELAY:.2f}s between calls")


def _note_success():
    """Ease the delay back down after a long stretch without throttling."""
    global RATE_LIMIT_DELAY, _consecutive_ok
    _consecutive_ok += 1
    if _consecutive_ok >= RATE_LIMIT_DECAY_AFTER and RATE_LIMIT_DELAY > RATE_LIMIT_MIN:
        previous = RATE_LIMIT_DELAY
        RATE_LIMIT_DELAY = max(RATE_LIMIT_DELAY * RATE_LIMIT_DECAY, RATE_LIMIT_MIN)
        _consecutive_ok = 0
        logger.info(f"{RATE_LIMIT_DECAY_AFTER} clean calls; speeding up "
                    f"{previous:.2f}s -> {RATE_LIMIT_DELAY:.2f}s between calls")


def _retry_after_seconds(response, fallback):
    """
    Honour a Retry-After header when the server sends one.

    Args:
        response: A requests.Response
        fallback: Seconds to wait if the header is absent or unparseable

    Returns:
        float: Seconds to sleep
    """
    value = response.headers.get("Retry-After")
    if value:
        try:
            return max(float(value), fallback)
        except (TypeError, ValueError):
            pass
    return fallback


def rate_limit_delay():
    """Enforce the current (adaptive) delay between API requests"""
    global last_api_call_time
    elapsed = time.time() - last_api_call_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    last_api_call_time = time.time()


def api_call_with_retry(url, json_payload, headers, timeout=30, max_attempts=RETRY_MAX_ATTEMPTS):
    """
    Make an API call with rate limiting and exponential backoff.

    Throttling responses do not count against `max_attempts`: they get their own,
    larger budget, because being told to slow down says nothing about whether the
    request itself is valid. Genuine client errors still fail fast.

    Args:
        url: Endpoint to POST to
        json_payload: Request body
        headers: Request headers
        timeout: Per-request timeout in seconds
        max_attempts: Budget for non-throttling failures

    Returns:
        requests.Response: The successful response

    Raises:
        Exception: When every attempt is exhausted
    """
    backoff = RETRY_INITIAL_BACKOFF
    last_error = None
    attempt = 0
    throttled = 0

    while attempt < max_attempts and throttled < RETRY_RATE_LIMIT_ATTEMPTS:
        try:
            rate_limit_delay()
            response = requests.post(url, json=json_payload, headers=headers, timeout=timeout)

            logger.debug(f"API response status: {response.status_code}")

            if _looks_rate_limited(response):
                throttled += 1
                _note_throttled()
                wait = _retry_after_seconds(response, backoff)
                logger.warning(f"Throttled ({response.status_code}) - rate-limit retry "
                               f"{throttled}/{RETRY_RATE_LIMIT_ATTEMPTS}. Waiting {wait:.1f}s...")
                time.sleep(wait)
                backoff = min(backoff * 2, RETRY_MAX_BACKOFF)
                continue

            attempt += 1

            if response.status_code != 200:
                logger.warning(f"Non-200 response ({response.status_code}) on attempt "
                               f"{attempt}/{max_attempts}: {response.text[:200]}")
                if response.status_code >= 500:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, RETRY_MAX_BACKOFF)
                    continue
                else:
                    response.raise_for_status()

            _note_success()
            return response

        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f"Timeout on attempt {attempt}/{max_attempts}")
            time.sleep(backoff)
            backoff = min(backoff * 2, RETRY_MAX_BACKOFF)

        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"Connection error on attempt {attempt}/{max_attempts}")
            time.sleep(backoff)
            backoff = min(backoff * 2, RETRY_MAX_BACKOFF)

        except requests.exceptions.RequestException as e:
            last_error = e
            logger.error(f"Request error on attempt {attempt}/{max_attempts}: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, RETRY_MAX_BACKOFF)

    if throttled >= RETRY_RATE_LIMIT_ATTEMPTS:
        raise Exception(f"Gave up after {throttled} consecutive rate-limit responses. "
                        f"Raise PROMPT_RATE_LIMIT and rerun - progress is checkpointed.")
    raise Exception(f"Max retries ({max_attempts}) exceeded. Last error: {last_error}")


def read_movies_desc(path=MOVIES_DESC_FILE):
    """Read movie descriptions from file with validation"""
    try:
        movies = pd.read_csv(path, delimiter='::',
                             names=['movieId', 'title', 'year', 'genres', 'description'],
                             engine='python', encoding='utf-8')
        logger.info(f"Loaded {len(movies)} movies from {path}")
        return movies
    except FileNotFoundError:
        raise FileNotFoundError(f"Movies description file not found: {path}")
    except Exception as e:
        raise Exception(f"Error reading movies file: {e}")


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------

def _headers():
    """Auth headers for the Purdue GenAI Studio REST API."""
    return {
        "Authorization": f"Bearer {PURDUE_API_KEY}",
        "Content-Type": "application/json",
    }


def _ask_llm(prompt):
    """
    Send one prompt and return the reply text, or None if the call failed.

    Args:
        prompt: The user-role prompt string

    Returns:
        str or None: The model's reply, stripped; None on any failure
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
    }

    try:
        response = api_call_with_retry(PURDUE_API_URL, json_payload=payload,
                                       headers=_headers(), timeout=30)
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None

    if response.status_code != 200:
        logger.error(f"API returned status {response.status_code}: {response.text[:300]}")
        return None

    try:
        result = response.json()
    except Exception as e:
        logger.error(f"Could not parse JSON response: {e}. Body: {response.text[:300]}")
        return None

    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        logger.error(f"Unexpected response structure: {str(result)[:300]}")
        return None


# ---------------------------------------------------------------------------
# Resumable prompt passes
# ---------------------------------------------------------------------------

def _run_prompt_pass(movies, output_file, build_prompt, label, force=False, retry_failed=True):
    """
    Generate one description per movie, resuming from whatever is already on disk.

    Only movies still missing a usable description are sent to the LLM, and the file
    is rewritten every CHECKPOINT_EVERY completions, so an interrupted run costs at
    most a handful of calls rather than the whole pass.

    Args:
        movies: DataFrame of movies (positional order defines entry order)
        output_file: Delimited descriptions file to read and write
        build_prompt: Callable taking a movie record dict, returning the prompt
        label: Noun used in log messages
        force: Ignore existing output and regenerate everything
        retry_failed: Retry entries previously written as [ERROR]

    Returns:
        list[str]: One description per movie, in movie order
    """
    records = movies.to_dict("records")
    total = len(records)

    existing = [] if force else load_entries(output_file)
    todo, descriptions = plan_work(existing, total, force=force,
                                   retry_failed=retry_failed, label=label)

    if not todo:
        logger.info(f"{label}: nothing to do, {output_file} is already complete")
        return descriptions

    logger.info(f"Generating {len(todo)} {label} via Purdue GenAI Studio "
                f"(~{len(todo) * RATE_LIMIT_DELAY / 3600:.1f}h at the current rate limit)...")
    completed = 0

    try:
        for position, idx in enumerate(todo, start=1):
            record = records[idx]
            logger.info(f"[{position}/{len(todo)}] movie {record['movieId']} ({label})")

            reply = _ask_llm(build_prompt(record))
            descriptions[idx] = reply if reply else "[ERROR]"
            completed += 1

            if completed % CHECKPOINT_EVERY == 0:
                save_entries(output_file, descriptions)
                logger.info(f"  checkpoint: {position}/{len(todo)} saved to {output_file}")
    finally:
        # Bank whatever finished, including on Ctrl-C or an unhandled error.
        save_entries(output_file, descriptions)

    failures = sum(1 for d in descriptions if is_placeholder(d))
    logger.info(f"{label} saved to {output_file} ({total - failures}/{total} usable)")
    if failures:
        logger.warning(f"{failures} {label} still failed; rerun to retry just those")

    return descriptions


def _basic_prompt(record):
    """Prompt asking the LLM to summarize a movie description."""
    return (
        "Summarize the following movie description in under 50 words. "
        "Keep all essential details. Do not add or remove details.\n\n"
        f"{record['description']}"
    )


def _rec_driven_prompt(record):
    """Prompt asking the LLM to pitch the movie to a prospective viewer."""
    return (
        "Use the movie description and state what you would say to someone to "
        "recommend the movie to them in 50 words or less.\n\n"
        f"{record['description']}"
    )


def generateBasicDescriptions(movies, output_file=BASIC_DESC_FILE, force=False, retry_failed=True):
    """Generate basic (summarization) descriptions, resuming from any partial run."""
    return _run_prompt_pass(movies, output_file, _basic_prompt, "basic descriptions",
                            force=force, retry_failed=retry_failed)


def generateRecDrivenDescriptions(movies, output_file=REC_DRIVEN_DESC_FILE, force=False,
                                  retry_failed=True):
    """Generate recommendation-driven descriptions, resuming from any partial run."""
    return _run_prompt_pass(movies, output_file, _rec_driven_prompt,
                            "recommendation-driven descriptions",
                            force=force, retry_failed=retry_failed)


def generateCombinedDescriptions(basic_descriptions, rec_descriptions, movies,
                                 output_file=MOVIES_ENHANCED_DESC_COMBINED, force=False,
                                 retry_failed=True):
    """
    Merge the two prompt outputs into one description per movie, resuming as needed.

    Note: model type E makes this pass unnecessary. It embeds each description
    separately and lets attention weight them per user, which avoids both this third
    LLM pass and the encoder truncation that concatenation causes.

    Args:
        basic_descriptions: List of basic summaries, in movie order
        rec_descriptions: List of recommendation pitches, in movie order
        movies: DataFrame of movies
        output_file: DAT file to read and write
        force: Ignore existing output and regenerate everything
        retry_failed: Retry entries previously written as [ERROR]

    Returns:
        list[str]: One combined description per movie, in movie order
    """
    records = movies.to_dict("records")
    total = len(records)

    existing_rows = {} if force else load_dat_rows(output_file)
    # .get() with no default: a movie absent from the file yields None ("never
    # attempted"), which is a different state from a present-but-empty row.
    existing = [existing_rows.get(str(r["movieId"])) for r in records]
    todo, combined = plan_work(existing, total, force=force, retry_failed=retry_failed,
                               label="combined descriptions", blank_is_pending=False)

    by_id = {str(r["movieId"]): c for r, c in zip(records, combined)}

    if not todo:
        logger.info(f"Combined descriptions: nothing to do, {output_file} is already complete")
        return combined

    logger.info(f"Generating {len(todo)} combined descriptions via Purdue GenAI Studio...")
    completed = 0

    try:
        for position, idx in enumerate(todo, start=1):
            record = records[idx]
            logger.info(f"[{position}/{len(todo)}] movie {record['movieId']} (combined)")

            basic = basic_descriptions[idx] if idx < len(basic_descriptions) else ""
            rec = rec_descriptions[idx] if idx < len(rec_descriptions) else ""
            basic = "" if is_placeholder(basic) else basic
            rec = "" if is_placeholder(rec) else rec

            prompt = (
                "Given the following two descriptions of a movie, create a single unified "
                "summary that combines the key aspects from both. The summary should be "
                "informative and engaging.\n\n"
                f"Summary perspective: {basic}\n\n"
                f"Recommendation perspective: {rec}\n\n"
                f"Original description: {record['description']}\n\n"
                "Provide only the combined summary, nothing else."
            )

            reply = _ask_llm(prompt)
            combined[idx] = reply if reply else "[ERROR]"
            by_id[str(record["movieId"])] = combined[idx]
            completed += 1

            if completed % CHECKPOINT_EVERY == 0:
                save_dat_rows(output_file, records, by_id)
                logger.info(f"  checkpoint: {position}/{len(todo)} saved to {output_file}")
    finally:
        save_dat_rows(output_file, records, by_id)

    failures = sum(1 for d in combined if is_placeholder(d))
    logger.info(f"Combined descriptions saved to {output_file} ({total - failures}/{total} usable)")
    if failures:
        logger.warning(f"{failures} combined descriptions still failed; rerun to retry just those")

    return combined


def generateDescEmbeddings(model, movies, output_file):
    """Generate embeddings for movie descriptions using SentenceTransformer"""
    try:
        descriptions = movies['description'].tolist()
        logger.info(f"Generating embeddings for {len(descriptions)} descriptions...")
        embeddings = model.encode(descriptions, convert_to_tensor=True)
        movies["embeddings"] = embeddings.tolist()
        movies.to_json(output_file, orient='records', force_ascii=False)
        logger.info(f"Movie embeddings saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance movie descriptions with an LLM. Resumes automatically: "
                    "movies that already have a usable description are skipped."
    )
    parser.add_argument('--force', action='store_true',
                        help='Ignore existing output files and regenerate everything')
    parser.add_argument('--no-retry-failed', action='store_true',
                        help='Leave previously-failed [ERROR] entries alone instead of retrying')
    parser.add_argument('--skip-combined', action='store_true',
                        help='Skip the combined pass (model type E does not need it)')
    args = parser.parse_args()

    retry_failed = not args.no_retry_failed

    logger.info("Starting LLM-based description enhancement via Purdue GenAI Studio...")
    start = time.time()

    movies = read_movies_desc()
    logger.info(f"Loaded {len(movies)} movies to enhance")

    basic_desc = generateBasicDescriptions(movies, force=args.force, retry_failed=retry_failed)
    recDriven_desc = generateRecDrivenDescriptions(movies, force=args.force, retry_failed=retry_failed)

    if args.skip_combined:
        logger.info("Skipping combined descriptions (--skip-combined)")
    else:
        generateCombinedDescriptions(basic_desc, recDriven_desc, movies,
                                     force=args.force, retry_failed=retry_failed)

    logger.info("Description enhancement complete. Generated files:")
    logger.info(f"  - Basic descriptions: {BASIC_DESC_FILE}")
    logger.info(f"  - Recommendation-driven descriptions: {REC_DRIVEN_DESC_FILE}")
    if not args.skip_combined:
        logger.info(f"  - Combined descriptions: {MOVIES_ENHANCED_DESC_COMBINED}")

    logger.info(f"Took {(time.time() - start):.2f} seconds.")
