import requests
import torch
import time
import os
import pandas as pd
import logging
import random
import numpy as np
from sentence_transformers import SentenceTransformer
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

# Rate limiting for API calls
RATE_LIMIT_DELAY = 5.0  # seconds between API calls (increased from 1.0)
RETRY_MAX_ATTEMPTS = 5  # increased attempts
RETRY_INITIAL_BACKOFF = 3.0  # increased from 2.0
RETRY_MAX_BACKOFF = 60.0  # increased from 30.0
last_api_call_time = 0.0

def rate_limit_delay():
    """Enforce rate limiting between API requests"""
    global last_api_call_time
    elapsed = time.time() - last_api_call_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    last_api_call_time = time.time()

def api_call_with_retry(url, json_payload, headers, timeout=30, max_attempts=RETRY_MAX_ATTEMPTS):
    """Make API call with exponential backoff retry logic and rate limiting"""
    backoff = RETRY_INITIAL_BACKOFF
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            rate_limit_delay()
            response = requests.post(url, json=json_payload, headers=headers, timeout=timeout)
            
            logger.debug(f"API response status: {response.status_code}")
            
            if response.status_code == 429:  # Rate limited
                logger.warning(f"RATE LIMITED (429) - Attempt {attempt+1}/{max_attempts}. Waiting {backoff}s before retry...")
                time.sleep(backoff)
                backoff = min(backoff * 2, RETRY_MAX_BACKOFF)
                continue
            
            if response.status_code != 200:
                logger.warning(f"Non-200 response ({response.status_code}) on attempt {attempt+1}/{max_attempts}: {response.text[:200]}")
                if response.status_code >= 500:
                    # Retry on server errors
                    time.sleep(backoff)
                    backoff = min(backoff * 2, RETRY_MAX_BACKOFF)
                    continue
                else:
                    # Don't retry on client errors
                    response.raise_for_status()
            
            return response
            
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f"Timeout on attempt {attempt+1}/{max_attempts}")
            time.sleep(backoff)
            backoff = min(backoff * 2, RETRY_MAX_BACKOFF)
            
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"Connection error on attempt {attempt+1}/{max_attempts}")
            time.sleep(backoff)
            backoff = min(backoff * 2, RETRY_MAX_BACKOFF)
            
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.error(f"Request error on attempt {attempt+1}/{max_attempts}: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, RETRY_MAX_BACKOFF)
    
    raise Exception(f"Max retries ({max_attempts}) exceeded. Last error: {last_error}")

def read_movies_desc(path=MOVIES_DESC_FILE):
   """Read movie descriptions from file with validation"""
   try:
       movies = pd.read_csv(path, delimiter='::', names=['movieId', 'title', 'year', 'genres','description'], engine='python', encoding='utf-8')
       logger.info(f"Loaded {len(movies)} movies from {path}")
       return movies
   except FileNotFoundError:
       raise FileNotFoundError(f"Movies description file not found: {path}")
   except Exception as e:
       raise Exception(f"Error reading movies file: {e}")

def generateBasicDescriptions(movies, output_file=BASIC_DESC_FILE):
    """Generate basic (summarization) descriptions using Purdue GenAI Studio"""
    mov_descriptions = [{"movieId": mov["movieId"], "description": mov["description"]} for mov in movies.to_dict('records')]
    basic_desc = []

    logger.info(f"Generating basic descriptions ({len(mov_descriptions)} movies) using Purdue GenAI Studio...")

    headers = {
        "Authorization": f"Bearer {PURDUE_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, desc_obj in enumerate(mov_descriptions):
                try:
                    logger.info(f"Processing movie {idx+1}/{len(mov_descriptions)}: {desc_obj['movieId']}")
                    
                    prompt = f"""Summarize the following movie description in under 50 words. Keep all essential details. Do not add or remove details.

{desc_obj['description']}"""

                    payload = {
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_NEW_TOKENS
                    }

                    response = api_call_with_retry(
                        PURDUE_API_URL,
                        json_payload=payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    logger.info(f"Got response for movie {desc_obj['movieId']}, status: {response.status_code}")

                    if response.status_code == 200:
                        logger.info(f"Received 200 response for movie {desc_obj['movieId']}, parsing JSON...")
                        try:
                            result = response.json()
                            logger.info(f"Successfully parsed JSON for movie {desc_obj['movieId']}")
                        except Exception as json_err:
                            logger.error(f"Failed to parse JSON for movie {desc_obj['movieId']}: {json_err}. Response text: {response.text[:500]}")
                            basic_desc.append("")
                            f.write("[ERROR]\n::\n")
                            continue
                        
                        # Validate response structure
                        if result and "choices" in result and result["choices"] and len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            if choice and "message" in choice and choice["message"] and "content" in choice["message"]:
                                summary = choice["message"]["content"].strip()
                                basic_desc.append(summary)
                                f.write(summary + "\n::\n")
                            else:
                                logger.error(f"Invalid response structure for movie {desc_obj['movieId']}: missing message or content - {result}")
                                basic_desc.append("")
                                f.write("[ERROR]\n::\n")
                        else:
                            logger.error(f"Invalid response structure for movie {desc_obj['movieId']}: missing choices - {result}")
                            basic_desc.append("")
                            f.write("[ERROR]\n::\n")
                    else:
                        logger.error(f"API returned status {response.status_code} for movie {desc_obj['movieId']}: {response.text[:500]}")
                        basic_desc.append("")
                        f.write("[ERROR]\n::\n")

                except Exception as e:
                    logger.error(f"Error processing movie {idx}: {e}")
                    basic_desc.append("")
                    f.write("[ERROR]\n::\n")
        
        logger.info(f"Basic descriptions saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating basic descriptions: {e}")
        raise

    return basic_desc

def generateRecDrivenDescriptions(movies, output_file=REC_DRIVEN_DESC_FILE):
    """Generate recommendation-driven descriptions using Purdue GenAI Studio"""
    mov_descriptions = [{"movieId": mov["movieId"], "description": mov["description"]} for mov in movies.to_dict('records')]
    recDriven_desc = []

    logger.info(f"Generating recommendation-driven descriptions ({len(mov_descriptions)} movies) using Purdue GenAI Studio...")

    headers = {
        "Authorization": f"Bearer {PURDUE_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, desc_obj in enumerate(mov_descriptions):
                try:
                    logger.info(f"Processing movie {idx+1}/{len(mov_descriptions)}: {desc_obj['movieId']}")
                    
                    prompt = f"""Use the movie description and state what you would say to someone to recommend the movie to them.

{desc_obj['description']}"""

                    payload = {
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_NEW_TOKENS
                    }

                    response = api_call_with_retry(
                        PURDUE_API_URL,
                        json_payload=payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    logger.info(f"Got response for movie {desc_obj['movieId']}, status: {response.status_code}")

                    if response.status_code == 200:
                        logger.info(f"Received 200 response for movie {desc_obj['movieId']}, parsing JSON...")
                        try:
                            result = response.json()
                            logger.info(f"Successfully parsed JSON for movie {desc_obj['movieId']}")
                        except Exception as json_err:
                            logger.error(f"Failed to parse JSON for movie {desc_obj['movieId']}: {json_err}. Response text: {response.text[:500]}")
                            recDriven_desc.append("")
                            f.write("[ERROR]\n::\n")
                            continue
                        
                        # Validate response structure
                        if result and "choices" in result and result["choices"] and len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            if choice and "message" in choice and choice["message"] and "content" in choice["message"]:
                                recommendation = choice["message"]["content"].strip()
                                recDriven_desc.append(recommendation)
                                f.write(recommendation + "\n::\n")
                            else:
                                logger.error(f"Invalid response structure for movie {desc_obj['movieId']}: missing message or content - {result}")
                                recDriven_desc.append("")
                                f.write("[ERROR]\n::\n")
                        else:
                            logger.error(f"Invalid response structure for movie {desc_obj['movieId']}: missing choices - {result}")
                            recDriven_desc.append("")
                            f.write("[ERROR]\n::\n")
                    else:
                        logger.error(f"API returned status {response.status_code} for movie {desc_obj['movieId']}: {response.text[:500]}")
                        recDriven_desc.append("")
                        f.write("[ERROR]\n::\n")

                except Exception as e:
                    logger.error(f"Error processing movie {idx}: {e}")
                    recDriven_desc.append("")
                    f.write("[ERROR]\n::\n")
        
        logger.info(f"Recommendation-driven descriptions saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating recommendation-driven descriptions: {e}")
        raise

    return recDriven_desc

def generateCombinedDescriptions(basic_descriptions, rec_descriptions, movies, output_file=MOVIES_ENHANCED_DESC_COMBINED):
    """Generate combined descriptions using both basic and recommendation-driven descriptions via Purdue GenAI Studio"""
    combined_desc = []

    logger.info(f"Generating combined descriptions ({len(movies)} movies) using Purdue GenAI Studio...")

    headers = {
        "Authorization": f"Bearer {PURDUE_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for seq_idx, (idx, row) in enumerate(movies.iterrows()):
                try:
                    movieId = row['movieId']
                    logger.info(f"Processing movie {seq_idx+1}/{len(movies)}: {movieId}")
                    
                    # Use sequential index, not DataFrame index!
                    basic = basic_descriptions[seq_idx] if seq_idx < len(basic_descriptions) and basic_descriptions[seq_idx] and basic_descriptions[seq_idx].strip() != "[ERROR]" else ""
                    rec = rec_descriptions[seq_idx] if seq_idx < len(rec_descriptions) and rec_descriptions[seq_idx] and rec_descriptions[seq_idx].strip() != "[ERROR]" else ""
                    original = row['description']
                    
                    prompt = f"""Given the following two descriptions of a movie, create a single unified summary that combines the key aspects from both. The summary should be informative and engaging.

Summary perspective: {basic}

Recommendation perspective: {rec}

Original description: {original}

Provide only the combined summary, nothing else."""

                    payload = {
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_NEW_TOKENS
                    }

                    response = api_call_with_retry(
                        PURDUE_API_URL,
                        json_payload=payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    logger.info(f"Got response for movie {movieId}, status: {response.status_code}")

                    if response.status_code == 200:
                        logger.info(f"Received 200 response for movie {movieId}, parsing JSON...")
                        try:
                            result = response.json()
                            logger.info(f"Successfully parsed JSON for movie {movieId}")
                        except Exception as json_err:
                            logger.error(f"Failed to parse JSON for movie {movieId}: {json_err}. Response text: {response.text[:500]}")
                            combined_desc.append("")
                            f.write(f"{row['movieId']}::{row['title']}::{row['year']}::{row['genres']}::[ERROR]\n")
                            continue
                        
                        # Validate response structure
                        if result and "choices" in result and result["choices"] and len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            if choice and "message" in choice and choice["message"] and "content" in choice["message"]:
                                combined = choice["message"]["content"].strip()
                                combined_desc.append(combined)
                                # Write in DAT format: movieId::title::year::genres::description
                                f.write(f"{row['movieId']}::{row['title']}::{row['year']}::{row['genres']}::{combined}\n")
                            else:
                                logger.error(f"Invalid response structure for movie {movieId}: missing message or content - {result}")
                                combined_desc.append("")
                                f.write(f"{row['movieId']}::{row['title']}::{row['year']}::{row['genres']}::[ERROR]\n")
                        else:
                            logger.error(f"Invalid response structure for movie {movieId}: missing choices - {result}")
                            combined_desc.append("")
                            f.write(f"{row['movieId']}::{row['title']}::{row['year']}::{row['genres']}::[ERROR]\n")
                    else:
                        logger.error(f"API returned status {response.status_code} for movie {movieId}: {response.text[:500]}")
                        combined_desc.append("")
                        f.write(f"{row['movieId']}::{row['title']}::{row['year']}::{row['genres']}::[ERROR]\n")

                except Exception as e:
                    logger.error(f"Error processing movie {seq_idx}: {e}")
                    combined_desc.append("")
                    f.write(f"{row['movieId']}::{row['title']}::{row['year']}::{row['genres']}::[ERROR]\n")
        
        logger.info(f"Combined descriptions saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating combined descriptions: {e}")
        raise

    return combined_desc


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
    logger.info("Starting LLM-based description enhancement via Purdue GenAI Studio...")
    start = time.time()
    
    # Read movie descriptions
    logger.info(f"Loading movie descriptions from {MOVIES_DESC_FILE}...")
    movies = read_movies_desc()
    logger.info(f"Loaded {len(movies)} movies to enhance")

    # Generate basic (summary) descriptions
    logger.info("Generating basic (summary) descriptions...")
    basic_desc = generateBasicDescriptions(movies)

    # Generate recommendation-driven descriptions
    logger.info("Generating recommendation-driven descriptions...")
    recDriven_desc = generateRecDrivenDescriptions(movies)

    # Generate combined descriptions
    logger.info("Generating combined descriptions...")
    combined_desc = generateCombinedDescriptions(basic_desc, recDriven_desc, movies)
    
    logger.info(f"All description enhancements completed successfully!")
    logger.info(f"Generated files:")
    logger.info(f"  - Basic descriptions: {BASIC_DESC_FILE}")
    logger.info(f"  - Recommendation-driven descriptions: {REC_DRIVEN_DESC_FILE}")
    logger.info(f"  - Combined descriptions: {MOVIES_ENHANCED_DESC_COMBINED}")
    
    end = time.time()
    logger.info(f"Description enhancement completed in {(end-start):.2f} seconds.")