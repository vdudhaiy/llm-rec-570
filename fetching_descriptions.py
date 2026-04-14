import requests
import pandas as pd
import re
import time
import random
import logging
from config import (
    TMDB_API_KEY_FILE, TMDB_INITIAL_DELAY, TMDB_MAX_RETRIES,
    TMDB_INITIAL_BACKOFF, TMDB_MAX_BACKOFF, RANDOM_SEED,
    MOVIES_FILE, MOVIES_DESC_FILE, VERBOSE
)

# Set seed for reproducibility
random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)

# Load and validate TMDB API key
try:
    with open(TMDB_API_KEY_FILE, "r") as f:
        TMDB_API_KEY = f.read().strip()
    if not TMDB_API_KEY:
        raise ValueError("TMDB_API_KEY is empty")
    logger.info(f"Loaded TMDB API key (first 10 chars: {TMDB_API_KEY[:10]}...)")
except FileNotFoundError:
    raise FileNotFoundError(f"TMDB API key file not found: {TMDB_API_KEY_FILE}")
except Exception as e:
    raise Exception(f"Error loading TMDB API key: {e}")

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
    try:
        with open(MOVIES_FILE, "r", encoding="utf-8") as f:
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
        return data.get("overview", "NOT FOUND")
    except Exception as e:
        logger.error(f"Error fetching description for movie ID {movie_id}: {e}")
        return "ERROR FETCHING"

def generate_descriptions():
    """Fetch descriptions for all movies from TMDB"""
    movies = read_movies()
    logger.info("Starting description fetching...")
    
    successful = 0
    failed = 0
    not_found = 0

    for i, mov in enumerate(movies):
        logger.info(f"Processing movie {i+1}/{len(movies)}: {mov['title']}")
        
        mov_id = get_tmdb_id(mov["title"])
        if mov_id is None:
            logger.warning(f"TMDB ID not found: {mov['title']}")
            mov["description"] = "NOT_FOUND"
            not_found += 1
            continue
        
        description = fetch_description(mov_id)
        mov["description"] = description
        
        if description in ["NOT_FOUND", "ERROR FETCHING"]:
            failed += 1
        else:
            successful += 1

    logger.info(f"Description fetching complete: {successful} successful, {failed} failed, {not_found} not found")
    
    logger.info(f"Writing to {MOVIES_DESC_FILE}...")
    
    try:
        with open(MOVIES_DESC_FILE, "w", encoding="utf-8") as f:
            for mov in movies:
                f.write(f"{mov['movieId']}::{mov['title']}::{mov['year']}::{mov['genres']}::{mov['description']}\n")
        logger.info(f"Descriptions saved to {MOVIES_DESC_FILE}")
    except Exception as e:
        logger.error(f"Error writing descriptions file: {e}")
        raise

    return

if __name__=="__main__":
    logger.info("Starting TMDB description fetching...")
    
    start = time.time()
    generate_descriptions()
    end = time.time()

    logger.info(f"TMDB description fetching took {(end-start):.2f} seconds.")
