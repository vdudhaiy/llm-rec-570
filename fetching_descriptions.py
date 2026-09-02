import argparse
import requests
import pandas as pd
import re
import time
import random
import logging

from checkpoint import load_dat_rows, save_dat_rows, is_placeholder, CHECKPOINT_EVERY

# MovieLens-1M is distributed in ISO-8859-1.
MOVIELENS_ENCODING = "latin-1"
from config import (
    TMDB_API_KEY, TMDB_INITIAL_DELAY, TMDB_MAX_RETRIES,
    TMDB_INITIAL_BACKOFF, TMDB_MAX_BACKOFF, RANDOM_SEED,
    MOVIES_FILE, MOVIES_DESC_FILE, VERBOSE
)

# Set seed for reproducibility
random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)

# config.py already resolves the key from TMDB_API_KEY (or TMDB_API_KEY_FILE).
# Importing this module must not explode when the key is absent - only calling
# generate_descriptions() actually needs it.
if not TMDB_API_KEY:
    logger.warning("TMDB_API_KEY is not set; generate_descriptions() will raise if called")
else:
    logger.info(f"Loaded TMDB API key (first 4 chars: {TMDB_API_KEY[:4]}...)")

INITIAL_DELAY = TMDB_INITIAL_DELAY
last_request_time = 0.0

def rate_limit():
    """Enforce rate limiting between API requests"""
    global last_request_time
    elapsed = time.time() - last_request_time
    if elapsed < INITIAL_DELAY:
        time.sleep(INITIAL_DELAY - elapsed)
    last_request_time = time.time()

def api_call_with_retry(url, max_retries=TMDB_MAX_RETRIES):
    """Make API call with exponential backoff retry logic"""
    backoff = TMDB_INITIAL_BACKOFF
    last_error = None
    
    for attempt in range(max_retries):
        try:
            rate_limit()
            response = requests.get(url, timeout=10)
            
            if response.status_code == 429:  # Rate limited
                logger.warning(f"Rate limited (429). Attempt {attempt+1}/{max_retries}")
                time.sleep(backoff)
                backoff = min(backoff * 2, TMDB_MAX_BACKOFF)
                continue
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}")
            time.sleep(backoff)
            backoff = min(backoff * 2, TMDB_MAX_BACKOFF)
            
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"Connection error on attempt {attempt+1}/{max_retries}")
            time.sleep(backoff)
            backoff = min(backoff * 2, TMDB_MAX_BACKOFF)
            
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.error(f"Request error on attempt {attempt+1}/{max_retries}: {e}")
            if response.status_code != 429:  # Don't retry on other 4xx errors
                raise
    
    raise Exception(f"Max retries ({max_retries}) exceeded. Last error: {last_error}")

def get_tmdb_id(title):
    """Fetch TMDB movie ID with error handling"""
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    
    try:
        response = api_call_with_retry(search_url)
        data = response.json()
        
        if "results" in data and data["results"]:
            return data["results"][0]["id"]
        else:
            logger.debug(f"Movie not found in TMDB: {title}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching TMDB ID for '{title}': {e}")
        return None

def read_movies():
    """Read and parse movies from MovieLens dataset with column name standardization"""
    movies = []
    # MovieLens-1M ships as ISO-8859-1, not UTF-8. Titles like
    # "Mis\xe9rables, Les (1995)" make a UTF-8 read raise UnicodeDecodeError.
    try:
        with open(MOVIES_FILE, "r", encoding=MOVIELENS_ENCODING) as f:
            data = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Movies file not found: {MOVIES_FILE}")
    
    for mov in data:
        try:
            parts = mov.strip().split("::")
            if len(parts) < 3:
                logger.warning(f"Invalid movie format (insufficient fields): {mov.strip()}")
                continue
            
            match = re.match(r"^(.*?)\s*(?:\([^)]+\))?\s*\((\d{4})\)$", parts[1])
            
            if match:
                title = match.group(1).strip()
                year = match.group(2).strip()
                # Use 'movieId' consistently
                movie_dict = {"movieId": parts[0], "title": title, "year": year, "genres": parts[2]}
                movies.append(movie_dict)
            else:
                logger.warning(f"Title parse failed: {parts[1]}")
        except Exception as e:
            logger.error(f"Error parsing movie line: {e}")
            continue
    
    logger.info(f"Successfully read {len(movies)} movies")
    return movies
        
def fetch_description(movie_id):
    """Fetch movie description from TMDB with error handling"""
    fetch_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    
    try:
        response = api_call_with_retry(fetch_url)
        data = response.json()
        overview = (data.get("overview") or "").strip()
        # Record an explicit marker rather than "": a blank would be
        # indistinguishable from a movie that was never looked up.
        return overview if overview else "NOT_FOUND"
    except Exception as e:
        logger.error(f"Error fetching description for movie ID {movie_id}: {e}")
        return "ERROR FETCHING"

def _require_api_key():
    if not TMDB_API_KEY:
        raise ValueError(
            "TMDB_API_KEY not found. Set it in .env (or point TMDB_API_KEY_FILE at a file) "
            "before fetching descriptions."
        )


def generate_descriptions(force=False, retry_failed=True):
    """
    Fetch descriptions for all movies from TMDB, resuming from any partial run.

    Two API calls per movie (title lookup, then details) over 3883 movies is around
    half an hour of network time. The previous version held everything in memory and
    wrote once at the very end, so an interruption at movie 3800 saved nothing. Now
    results are checkpointed to disk as they arrive and a rerun fetches only what is
    still missing.

    Args:
        force: Ignore the existing file and refetch every movie
        retry_failed: Also refetch movies previously recorded as NOT_FOUND/ERROR

    Returns:
        dict: movieId (str) -> description
    """
    _require_api_key()
    movies = read_movies()
    total = len(movies)

    descriptions = {} if force else load_dat_rows(MOVIES_DESC_FILE)
    if force:
        logger.info(f"--force: refetching all {total} descriptions")

    todo = []
    retrying = 0
    for mov in movies:
        mid = str(mov["movieId"])
        if mid not in descriptions:
            # No row at all: this movie has never been looked up.
            todo.append(mov)
        elif is_placeholder(descriptions[mid]) and retry_failed:
            # A row exists but holds nothing usable - a completed attempt that
            # failed, so only redo it when explicitly asked.
            todo.append(mov)
            retrying += 1

    if not todo:
        logger.info(f"Nothing to fetch: {MOVIES_DESC_FILE} already covers all {total} movies")
        return descriptions

    done = total - len(todo)
    if done:
        msg = f"Resuming: {done}/{total} descriptions already fetched, {len(todo)} to go"
        if retrying:
            msg += f" (including {retrying} previously-failed entries being retried)"
        logger.info(msg)
    else:
        logger.info(f"Fetching all {total} descriptions from TMDB...")

    successful = 0
    failed = 0
    completed = 0

    try:
        for position, mov in enumerate(todo, start=1):
            mid = str(mov["movieId"])
            logger.info(f"[{position}/{len(todo)}] {mov['title']}")

            tmdb_id = get_tmdb_id(mov["title"])
            if tmdb_id is None:
                logger.warning(f"TMDB ID not found: {mov['title']}")
                descriptions[mid] = "NOT_FOUND"
                failed += 1
            else:
                description = fetch_description(tmdb_id)
                descriptions[mid] = description
                if is_placeholder(description):
                    failed += 1
                else:
                    successful += 1

            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                save_dat_rows(MOVIES_DESC_FILE, movies, descriptions)
                logger.info(f"  checkpoint: {position}/{len(todo)} saved to {MOVIES_DESC_FILE}")
    finally:
        # Bank whatever finished, including on Ctrl-C or an unhandled error.
        save_dat_rows(MOVIES_DESC_FILE, movies, descriptions)

    remaining = sum(1 for mov in movies if is_placeholder(descriptions.get(str(mov["movieId"]))))
    logger.info(f"Fetch complete: {successful} succeeded, {failed} failed this run")
    logger.info(f"{MOVIES_DESC_FILE} now has {total - remaining}/{total} usable descriptions")
    if remaining:
        logger.warning(f"{remaining} movies still have no description; rerun to retry just those")

    return descriptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch movie synopses from TMDB. Resumes automatically: movies "
                    "that already have a description are skipped."
    )
    parser.add_argument('--force', action='store_true',
                        help='Ignore the existing file and refetch every movie')
    parser.add_argument('--no-retry-failed', action='store_true',
                        help='Leave previously-failed entries alone instead of retrying')
    args = parser.parse_args()

    logger.info("Starting TMDB description fetching...")

    start = time.time()
    generate_descriptions(force=args.force, retry_failed=not args.no_retry_failed)
    end = time.time()

    logger.info(f"TMDB description fetching took {(end-start):.2f} seconds.")
