"""
Main pipeline orchestrator for LLM-Rec Recommendation System
Orchestrates: data fetching -> description enhancement -> embedding generation -> model training
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path

# Import configuration
from config import (
    MOVIES_DESC_FILE, MOVIES_FILE, RATINGS_FILE,
    BASIC_DESC_FILE, REC_DRIVEN_DESC_FILE,
    MOVIES_ENHANCED_DESC_BASIC, 
    MOVIES_ENHANCED_DESC_REC_DRIVEN, MOVIES_ENHANCED_DESC_COMBINED,
    EMBEDDINGS_FILE, EMBEDDINGS_BASIC, EMBEDDINGS_REC_DRIVEN, EMBEDDINGS_COMBINED,
    EMBEDDINGS_MULTIVIEW, MULTIVIEW_SOURCES,
    DESCRIPTIONS_FILE_MAP, EMBEDDINGS_FILE_MAP,
    EMBEDDING_MODEL, EMBEDDING_DIM, MODEL_TYPE, RANDOM_SEED,
    VERBOSE, TRAIN_EPOCHS, get_device, configure_backend, describe_device,
    OUTPUT_DIR, output_path
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline controller"""
    
    def __init__(self, model_type=None, force=False, retry_failed=False):
        """
        Initialize pipeline with configuration.

        Args:
            model_type: Override config model type (A-E)
            force: Regenerate every artefact instead of reusing what is on disk
            retry_failed: Re-issue API calls for entries that previously failed.
                          Off by default so a rerun costs nothing when the only
                          gaps are movies the upstream API has no data for.
        """
        self.model_type = model_type or MODEL_TYPE
        self.force = force
        self.retry_failed = retry_failed
        self._encoder = None
        self.device = get_device()
        configure_backend(self.device)

        logger.info("=" * 70)
        logger.info("LLM-Rec Pipeline Initialized")
        logger.info("=" * 70)
        logger.info(f"Model Type: {self.model_type}")
        logger.info(f"Device: {describe_device(self.device)}")
        if self.force:
            logger.info("--force: every step will regenerate, ignoring existing files")
        else:
            logger.info("Pipeline will reuse existing output files and resume partial work")
            if self.retry_failed:
                logger.info("--retry-failed: previously-failed entries will be requested again")

    @property
    def encoder(self):
        """SentenceTransformer, loaded on first use and placed on the GPU."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence encoder {EMBEDDING_MODEL} on {self.device}...")
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, device=str(self.device))
        return self._encoder
    
    def step_1_fetch_descriptions(self):
        """Step 1: Fetch movie descriptions from TMDB"""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Fetching Movie Descriptions from TMDB")
        logger.info("=" * 70)
        
        # generate_descriptions() resumes on its own, so calling it when the file
        # exists is safe: it fetches only what is missing. Skip the call entirely
        # only when the file is already complete.
        if not self.force and self._descriptions_complete(MOVIES_DESC_FILE):
            logger.info(f"✓ Movies description file already complete: {MOVIES_DESC_FILE}")
            logger.info("Skipping TMDB fetch")
            return
        
        try:
            logger.info("Starting TMDB description fetch...")
            # Importing the module is not enough - its work lives behind a
            # `if __name__ == "__main__"` guard, so call the function directly.
            from fetching_descriptions import generate_descriptions
            generate_descriptions(force=self.force, retry_failed=self.retry_failed)
            logger.info("✓ TMDB descriptions fetched successfully")
        except Exception as e:
            logger.error(f"Error fetching TMDB descriptions: {e}")
            raise
    
    def step_2_generate_enhanced_descriptions(self):
        """Step 2: Generate enhanced descriptions using Purdue GenAI"""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Generating Enhanced Descriptions")
        logger.info("=" * 70)
        
        if not os.path.exists(MOVIES_DESC_FILE):
            raise FileNotFoundError(
                f"Movies description file not found: {MOVIES_DESC_FILE}\n"
                f"Please run Step 1 first"
            )
        
        # These passes resume internally, so "the file exists" is not enough: a file
        # holding 441 of 3883 entries exists but is nowhere near done. Only skip a
        # pass when its output actually covers every movie.
        movie_count = self._movie_count()
        basic_exists = not self.force and self._entries_complete(BASIC_DESC_FILE, movie_count)
        rec_exists = not self.force and self._entries_complete(REC_DRIVEN_DESC_FILE, movie_count)
        combined_exists = not self.force and self._descriptions_complete(
            MOVIES_ENHANCED_DESC_COMBINED, movie_count
        )

        if basic_exists and rec_exists and combined_exists:
            logger.info(f"✓ Basic descriptions already exist: {BASIC_DESC_FILE}")
            logger.info(f"✓ Recommendation-driven descriptions already exist: {REC_DRIVEN_DESC_FILE}")
            logger.info(f"✓ Combined descriptions already exist: {MOVIES_ENHANCED_DESC_COMBINED}")
            logger.info("Skipping enhanced description generation")
            return
        
        try:
            from prompting import (
                read_movies_desc, generateBasicDescriptions,
                generateRecDrivenDescriptions, generateCombinedDescriptions
            )
            
            logger.info(f"Loading movies from {MOVIES_DESC_FILE}...")
            movies = read_movies_desc()
            
            # Generate basic descriptions
            if not basic_exists:
                logger.info("Generating basic (summarization) descriptions...")
                generateBasicDescriptions(movies, output_file=BASIC_DESC_FILE,
                                          force=self.force, retry_failed=self.retry_failed)
                logger.info(f"✓ Basic descriptions saved to {BASIC_DESC_FILE}")
            else:
                logger.info(f"✓ Basic descriptions already exist: {BASIC_DESC_FILE}")
            
            # Generate recommendation-driven descriptions
            if not rec_exists:
                logger.info("Generating recommendation-driven descriptions...")
                generateRecDrivenDescriptions(movies, output_file=REC_DRIVEN_DESC_FILE,
                                              force=self.force, retry_failed=self.retry_failed)
                logger.info(f"✓ Recommendation-driven descriptions saved to {REC_DRIVEN_DESC_FILE}")
            else:
                logger.info(f"✓ Recommendation-driven descriptions already exist: {REC_DRIVEN_DESC_FILE}")
            
            # Generate combined descriptions
            if not combined_exists:
                logger.info("Generating combined descriptions...")
                # Read the description files
                basic_descriptions = self._read_descriptions_file(BASIC_DESC_FILE)
                rec_descriptions = self._read_descriptions_file(REC_DRIVEN_DESC_FILE)
                
                # Call generateCombinedDescriptions
                generateCombinedDescriptions(
                    basic_descriptions, rec_descriptions, movies,
                    output_file=MOVIES_ENHANCED_DESC_COMBINED,
                    force=self.force, retry_failed=self.retry_failed
                )
                logger.info(f"✓ Combined descriptions saved to {MOVIES_ENHANCED_DESC_COMBINED}")
            else:
                logger.info(f"✓ Combined descriptions already exist: {MOVIES_ENHANCED_DESC_COMBINED}")
            
            logger.info("✓ Enhanced description generation complete")
            
        except Exception as e:
            logger.error(f"Error generating enhanced descriptions: {e}")
            raise
    
    def step_3_generate_embeddings(self):
        """Step 3: Generate embeddings for all description types"""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Generating Embeddings")
        logger.info("=" * 70)
        
        if not os.path.exists(MOVIES_DESC_FILE):
            raise FileNotFoundError(f"Movies description file not found: {MOVIES_DESC_FILE}")
        
        # Check which embeddings files already exist
        # An embeddings file is reusable only if it is newer than the descriptions
        # it was built from. Step 2 rewriting a descriptions file invalidates it.
        embeddings_exist = {
            'A': self._is_current(EMBEDDINGS_FILE, MOVIES_DESC_FILE),
            'B': self._is_current(EMBEDDINGS_BASIC, BASIC_DESC_FILE, MOVIES_DESC_FILE),
            'C': self._is_current(EMBEDDINGS_REC_DRIVEN, REC_DRIVEN_DESC_FILE, MOVIES_DESC_FILE),
            'D': self._is_current(EMBEDDINGS_COMBINED, MOVIES_ENHANCED_DESC_COMBINED),
            # Type E is deliberately absent here: its inputs are the Type A-D files
            # regenerated further down, so its staleness cannot be judged until they
            # exist. _generate_multiview_embeddings() makes that call itself.
        }
        
        # Skip if all required embeddings exist
        all_exist = all(embeddings_exist.values()) and self._is_current(
            EMBEDDINGS_MULTIVIEW, *[f for _, f in MULTIVIEW_SOURCES]
        )
        if all_exist:
            logger.info("✓ All embedding files already exist")
            for model_type, exists in embeddings_exist.items():
                if model_type == 'A':
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_FILE}")
                elif model_type == 'B':
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_BASIC}")
                elif model_type == 'C':
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_REC_DRIVEN}")
                elif model_type == 'D':
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_COMBINED}")
                else:
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_MULTIVIEW}")
            logger.info("Skipping embedding generation")
            return
        
        try:
            logger.info(f"Loading movies from {MOVIES_DESC_FILE}...")
            movies = pd.read_csv(
                MOVIES_DESC_FILE,
                delimiter='::',
                names=['movieId', 'title', 'year', 'genres', 'description'],
                engine='python'
            )
            
            # Type A: Embeddings from original descriptions
            if not embeddings_exist['A']:
                logger.info(f"\n--- Type A: Original Descriptions ---")
                logger.info(f"Using original descriptions from {MOVIES_DESC_FILE}")
                self._generate_embeddings_file(
                    movies, 'description', EMBEDDINGS_FILE,
                    "original descriptions"
                )
            else:
                logger.info(f"✓ Type A embeddings already exist: {EMBEDDINGS_FILE}")
            
            # Type B: Embeddings from basic descriptions only
            if not embeddings_exist['B']:
                logger.info(f"\n--- Type B: Basic Descriptions Only ---")
                if os.path.exists(BASIC_DESC_FILE):
                    basic_descriptions = self._align_descriptions(
                        self._read_descriptions_file(BASIC_DESC_FILE), movies, "Basic descriptions"
                    )
                    movies_basic = movies.copy()
                    movies_basic['description'] = basic_descriptions
                    self._save_enhanced_descriptions_file(
                        movies, basic_descriptions, MOVIES_ENHANCED_DESC_BASIC,
                        "basic descriptions"
                    )
                    self._generate_embeddings_file(
                        movies_basic, 'description', EMBEDDINGS_BASIC,
                        "basic descriptions"
                    )
                else:
                    logger.warning(f"Basic descriptions file not found: {BASIC_DESC_FILE}")
            else:
                logger.info(f"✓ Type B embeddings already exist: {EMBEDDINGS_BASIC}")
            
            # Type C: Embeddings from recommendation-driven descriptions only
            if not embeddings_exist['C']:
                logger.info(f"\n--- Type C: Recommendation-Driven Descriptions Only ---")
                if os.path.exists(REC_DRIVEN_DESC_FILE):
                    rec_descriptions = self._align_descriptions(
                        self._read_descriptions_file(REC_DRIVEN_DESC_FILE), movies,
                        "Recommendation-driven descriptions"
                    )
                    movies_rec = movies.copy()
                    movies_rec['description'] = rec_descriptions
                    self._save_enhanced_descriptions_file(
                        movies, rec_descriptions, MOVIES_ENHANCED_DESC_REC_DRIVEN,
                        "recommendation-driven descriptions"
                    )
                    self._generate_embeddings_file(
                        movies_rec, 'description', EMBEDDINGS_REC_DRIVEN,
                        "recommendation-driven descriptions"
                    )
                else:
                    logger.warning(f"Recommendation-driven descriptions file not found: {REC_DRIVEN_DESC_FILE}")
            else:
                logger.info(f"✓ Type C embeddings already exist: {EMBEDDINGS_REC_DRIVEN}")
            
            # Type D: Embeddings from combined descriptions
            if not embeddings_exist['D']:
                logger.info(f"\n--- Type D: Combined Descriptions ---")
                if os.path.exists(MOVIES_ENHANCED_DESC_COMBINED):
                    combined_movies = pd.read_csv(
                        MOVIES_ENHANCED_DESC_COMBINED,
                        delimiter='::',
                        names=['movieId', 'title', 'year', 'genres', 'description'],
                        engine='python'
                    )
                    if len(combined_movies) == len(movies):
                        self._generate_embeddings_file(
                            combined_movies, 'description', EMBEDDINGS_COMBINED,
                            "combined descriptions"
                        )
                    else:
                        logger.warning(f"Combined descriptions count mismatch: {len(combined_movies)} vs {len(movies)}")
                else:
                    logger.warning(f"Combined descriptions file not found: {MOVIES_ENHANCED_DESC_COMBINED}")
            else:
                logger.info(f"✓ Type D embeddings already exist: {EMBEDDINGS_COMBINED}")
            
            # Type E: stack the per-view embeddings just generated.
            logger.info(f"\n--- Type E: Multi-View (attention over each description) ---")
            self._generate_multiview_embeddings()

            logger.info("\n✓ Embedding generation complete")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _generate_multiview_embeddings(self):
        """
        Stack the single-view embedding files into one multi-view file (Type E).

        Types B/C/D concatenate several descriptions into a single string and embed
        it once. That has two problems: the sentence encoder truncates at 128 tokens,
        so a concatenation of three descriptions loses the tail; and the attention
        layer receives a sequence of length 1, where softmax is always 1.0 and no
        choice is possible.

        Here each description keeps its own embedding, so a movie becomes several
        vectors and the attention layer learns, per user, which description type to
        weight. No new LLM calls are needed - this reuses the per-view embeddings
        Step 3 already produced.
        """
        # Existence is not enough: this file is a stack of the per-view embeddings,
        # so regenerating any of them invalidates it. Because those are rebuilt
        # earlier in this same Step 3 call, the check has to happen here rather than
        # in the upfront survey - by now the inputs' timestamps are final.
        if self._is_current(EMBEDDINGS_MULTIVIEW, *[f for _, f in MULTIVIEW_SOURCES]):
            logger.info(f"Multi-view embeddings are current: {EMBEDDINGS_MULTIVIEW}")
            return

        available = [(name, path) for name, path in MULTIVIEW_SOURCES if os.path.exists(path)]
        missing = [name for name, path in MULTIVIEW_SOURCES if not os.path.exists(path)]

        if missing:
            logger.warning(
                f"Multi-view: skipping view(s) {', '.join(missing)} - embeddings file absent. "
                f"Run Step 2 for the corresponding descriptions to include them."
            )
        if len(available) < 2:
            logger.warning(
                f"Multi-view needs at least 2 views, found {len(available)}. "
                f"Skipping {EMBEDDINGS_MULTIVIEW}; with a single view the attention "
                f"layer has nothing to choose between."
            )
            return

        view_names = [name for name, _ in available]
        logger.info(f"Building multi-view embeddings from {len(view_names)} views: "
                    f"{', '.join(view_names)}")

        per_view = []
        for name, path in available:
            with open(path, "r", encoding="utf-8") as f:
                per_view.append(json.load(f))

        # Only movies present in every view can be stacked into a rectangular tensor.
        shared = set(per_view[0])
        for view in per_view[1:]:
            shared &= set(view)
        dropped = len(per_view[0]) - len(shared)
        if dropped:
            logger.warning(f"Multi-view: {dropped} movie(s) missing from at least one view, dropped")

        stacked = {mid: [view[mid] for view in per_view] for mid in shared}

        with open(EMBEDDINGS_MULTIVIEW, "w", encoding="utf-8") as f:
            json.dump({"views": view_names, "embeddings": stacked}, f)

        logger.info(f"Saved {len(stacked)} multi-view embeddings "
                    f"({len(view_names)} views each) to {EMBEDDINGS_MULTIVIEW}")

    def _is_current(self, output_file, *source_files):
        """
        True if `output_file` exists and is newer than every source it derives from.

        Embeddings are a function of the description text. When Step 2 rewrites a
        descriptions file, any embeddings built from the older version are stale -
        they exist, but they encode text that is no longer there. Checking only for
        existence would silently train the model on the previous run's data.

        Args:
            output_file: The derived artefact
            source_files: Inputs it was generated from; missing ones are ignored

        Returns:
            bool: True when the output can be reused as-is
        """
        if self.force or not os.path.exists(output_file):
            return False

        out_mtime = os.path.getmtime(output_file)
        for src in source_files:
            if os.path.exists(src) and os.path.getmtime(src) > out_mtime:
                logger.info(f"{output_file} is older than {src} - regenerating")
                return False
        return True

    def _movie_count(self):
        """How many movies the finished description files should cover."""
        with open(MOVIES_DESC_FILE, encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())

    def _descriptions_complete(self, path, expected=None):
        """
        True if a DAT file exists and every movie in it has a real description.

        Args:
            path: Path to the .dat file
            expected: Row count required, or None to accept whatever is present

        Returns:
            bool: True when the file needs no further work
        """
        from checkpoint import load_dat_rows, is_placeholder

        if not os.path.exists(path):
            return False
        rows = load_dat_rows(path)
        if not rows:
            return False
        if expected is not None and len(rows) < expected:
            logger.info(f"{path} covers {len(rows)}/{expected} movies - will resume")
            return False
        missing = sum(1 for v in rows.values() if is_placeholder(v))
        if missing:
            if self.retry_failed:
                logger.info(f"{path} has {missing} failed entries - will retry them")
                return False
            logger.info(f"{path} has {missing} entries with no description "
                        f"(pass --retry-failed to request them again)")
        return True

    def _entries_complete(self, path, expected):
        """
        True if a delimited descriptions file covers every movie with real content.

        Args:
            path: Path to the delimited descriptions file
            expected: Number of entries required

        Returns:
            bool: True when the file needs no further work
        """
        from checkpoint import load_entries, is_blank, is_failure

        if not os.path.exists(path):
            return False
        entries = load_entries(path)
        if len(entries) < expected:
            logger.info(f"{path} has {len(entries)}/{expected} entries - will resume")
            return False

        # Blank slots are reserved-but-never-generated, e.g. from an interrupted
        # run. They always count as outstanding work, regardless of --retry-failed.
        blank = sum(1 for e in entries if is_blank(e))
        if blank:
            logger.info(f"{path} has {blank} entries that were never generated - will resume")
            return False

        failed = sum(1 for e in entries if is_failure(e))
        if failed:
            if self.retry_failed:
                logger.info(f"{path} has {failed} failed entries - will retry them")
                return False
            logger.info(f"{path} has {failed} failed entries "
                        f"(pass --retry-failed to request them again)")
        return True

    def _read_descriptions_file(self, filepath):
        """
        Read descriptions from file with '::' delimiters.
        
        Args:
            filepath: Path to descriptions file
            
        Returns:
            List of descriptions
        """
        descriptions = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by '::' delimiter
            parts = content.split('\n::\n')
            for part in parts:
                if part.strip():  # Skip empty parts
                    descriptions.append(part.strip())
            
            logger.info(f"Read {len(descriptions)} descriptions from {filepath}")
            return descriptions
        except Exception as e:
            logger.error(f"Error reading descriptions file {filepath}: {e}")
            raise
    
    def _align_descriptions(self, descriptions, movies, label):
        """
        Line up a descriptions file with the movie table.

        The description files are positional - line N belongs to movie N - so a
        run of prompting.py that stopped early leaves a short file. Rather than
        silently skipping the whole model type (which is what used to happen),
        fill the gap with each movie's original TMDB description and say how much
        of the file is actually LLM-enhanced.

        Args:
            descriptions: List of descriptions read from the file
            movies: DataFrame with the canonical movie order and 'description'
            label: Human-readable description type, for logging

        Returns:
            List of descriptions the same length as movies
        """
        originals = movies['description'].tolist()

        if len(descriptions) > len(originals):
            raise ValueError(
                f"{label}: got {len(descriptions)} descriptions for {len(originals)} movies. "
                f"The file is out of sync with {MOVIES_DESC_FILE}; regenerate it."
            )

        aligned = []
        enhanced = 0
        for idx, original in enumerate(originals):
            candidate = descriptions[idx].strip() if idx < len(descriptions) else ""
            if candidate and candidate != "[ERROR]":
                aligned.append(candidate)
                enhanced += 1
            else:
                aligned.append(original)

        coverage = 100.0 * enhanced / max(len(originals), 1)
        if enhanced < len(originals):
            logger.warning(
                f"{label}: only {enhanced}/{len(originals)} movies ({coverage:.1f}%) have an "
                f"LLM-enhanced description. The rest fall back to the original TMDB text, so "
                f"this variant is only partially enhanced - rerun prompting.py for a full comparison."
            )
        else:
            logger.info(f"{label}: all {enhanced} movies enhanced")

        return aligned

    def _save_enhanced_descriptions_file(self, movies_df, descriptions, output_file, description_type):
        """
        Save enhanced descriptions to a DAT file.
        
        Args:
            movies_df: DataFrame with movie metadata
            descriptions: List of enhanced descriptions
            output_file: Output file path
            description_type: Type description (for logging)
        """
        if self._is_current(output_file, BASIC_DESC_FILE, REC_DRIVEN_DESC_FILE, MOVIES_DESC_FILE):
            logger.info(f"Enhanced descriptions file is current: {output_file}")
            return
        
        try:
            logger.info(f"Saving {len(descriptions)} {description_type} to {output_file}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for idx, (movie_id, title, year, genres) in enumerate(zip(
                    movies_df['movieId'], 
                    movies_df['title'],
                    movies_df['year'],
                    movies_df['genres']
                )):
                    description = descriptions[idx] if idx < len(descriptions) else ""
                    # Format: movieId::title::year::genres::description
                    line = f"{movie_id}::{title}::{year}::{genres}::{description}\n"
                    f.write(line)
            
            logger.info(f"✓ Saved {len(descriptions)} enhanced descriptions to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced descriptions file: {e}")
            raise
    
    def _generate_embeddings_file(self, movies_df, desc_column, output_file, description_type):
        """
        Generate and save embeddings for a set of descriptions.
        
        Args:
            movies_df: DataFrame with movie data
            desc_column: Column name containing descriptions
            output_file: Output JSON file for embeddings
            description_type: Description of what type (for logging)
        """
        # The caller has already decided this file needs (re)building; if it is
        # present it is stale, so replace it rather than returning early.
        if os.path.exists(output_file):
            logger.info(f"Replacing stale embeddings: {output_file}")
        
        try:
            logger.info(f"Generating embeddings for {len(movies_df)} movies from {description_type}...")
            
            # Build every input string first, then encode in one batched GPU pass.
            # Encoding one movie per call left the GPU idle almost the whole time.
            texts = [
                (
                    f"Movie title: {title} ({year}). "
                    f"Genres: {genres}. "
                    f"Description: {description}"
                ).strip()
                for title, year, genres, description in zip(
                    movies_df['title'], movies_df['year'],
                    movies_df['genres'], movies_df[desc_column]
                )
            ]

            encoded = self.encoder.encode(
                texts, batch_size=256, show_progress_bar=VERBOSE, convert_to_numpy=True
            )
            embeddings = {
                int(movie_id): vec.tolist()
                for movie_id, vec in zip(movies_df['movieId'], encoded)
            }
            
            # Save embeddings
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings file: {e}")
            raise
    
    def step_4_create_datasets(self):
        """Step 4: Create and validate datasets for training"""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Creating and Validating Datasets")
        logger.info("=" * 70)
        
        # Verify required files exist
        if not os.path.exists(RATINGS_FILE):
            raise FileNotFoundError(f"Ratings file not found: {RATINGS_FILE}")
        
        if not os.path.exists(MOVIES_DESC_FILE):
            raise FileNotFoundError(
                f"Movies description file not found: {MOVIES_DESC_FILE}\n"
                f"Please run Step 1 first"
            )
        
        try:
            from data_processing import createDataset
            
            logger.info(f"Loading datasets...")
            logger.info(f"  Ratings file: {RATINGS_FILE}")
            logger.info(f"  Movies file: {MOVIES_DESC_FILE}")
            
            # Create datasets and dataloaders
            train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = createDataset()
            
            logger.info(f"✓ Datasets created and validated:")
            logger.info(f"  Training set: {len(train_dataset)} samples")
            logger.info(f"  Validation set: {len(val_dataset)} samples")
            logger.info(f"  Test set: {len(test_dataset)} samples")
            logger.info(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")
            
        except Exception as e:
            logger.error(f"Error creating datasets: {e}")
            raise
    
    def step_5_train_and_test_model(self, epochs=TRAIN_EPOCHS, results_file=None):
        """Step 5: Train the recommendation model and evaluate on test set"""
        logger.info("\n" + "=" * 70)
        logger.info(f"STEP 5: Training and Testing Model (Type {self.model_type})")
        logger.info("=" * 70)
        
        # Verify required embeddings file exists for model type
        embeddings_file_map = {
            'A': EMBEDDINGS_FILE,
            'B': EMBEDDINGS_BASIC,
            'C': EMBEDDINGS_REC_DRIVEN,
            'D': EMBEDDINGS_COMBINED,
            'E': EMBEDDINGS_MULTIVIEW,
        }
        
        required_embeddings = embeddings_file_map[self.model_type]
        
        if not os.path.exists(required_embeddings):
            logger.warning(f"Required embeddings file not found: {required_embeddings}")
            logger.warning("Running Step 3 (Embedding Generation) first...")
            self.step_3_generate_embeddings()
        
        if not os.path.exists(RATINGS_FILE) or not os.path.exists(MOVIES_DESC_FILE):
            logger.warning("Dataset files missing, running Step 4 first...")
            self.step_4_create_datasets()
        
        results_file = results_file or f"test_results_{self.model_type}.json"

        try:
            logger.info(f"Starting model training with type {self.model_type}...")
            # Same trap as Step 1: `import training` runs nothing. Call run()
            # explicitly and pass the model type chosen on the command line,
            # otherwise --model-type is silently ignored.
            import training
            results = training.run(model_type=self.model_type, epochs=epochs,
                                   results_file=results_file)
            logger.info("✓ Model training and evaluation complete")
            logger.info(f"✓ Test results saved to {output_path(results_file)}")
            return results

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def run(self, epochs=TRAIN_EPOCHS):
        """Run the complete pipeline"""
        try:
            logger.info("Starting LLM-Rec Pipeline")
            
            self.step_1_fetch_descriptions()
            self.step_2_generate_enhanced_descriptions()
            self.step_3_generate_embeddings()
            self.step_4_create_datasets()
            self.step_5_train_and_test_model(epochs=epochs)
            
            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETE ✓")
            logger.info("=" * 70)
            logger.info("All steps completed successfully!")
            logger.info(f"Check {output_path(f'test_results_{self.model_type}.json')} "
                        f"for evaluation metrics")
            
        except Exception as e:
            logger.error(f"\nPipeline failed: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LLM-Rec Pipeline: Fetch descriptions, enhance with LLM, generate embeddings, create datasets, train & test model. "
                    "Automatically skips completed steps based on output files."
    )
    
    parser.add_argument(
        '--model-type', '-m',
        choices=['A', 'B', 'C', 'D', 'E'],
        help='Model type to train: A=SimpleCF, B=LLMRec+Basic, C=LLMRec+RecDriven, '
             'D=LLMRec+Combined, E=LLMRec+MultiView attention'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Regenerate everything from scratch instead of reusing existing files. '
             'Without this, completed work is skipped and partial work is resumed.'
    )

    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='Re-issue API calls for entries that previously failed. Off by default, '
             'since some movies simply have no upstream data and always fail.'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int, default=TRAIN_EPOCHS,
        help=f'Number of training epochs (default: {TRAIN_EPOCHS})'
    )

    parser.add_argument(
        '--step',
        choices=['1', '2', '3', '4', '5'],
        help='Run only a specific pipeline step'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Pipeline(model_type=args.model_type, force=args.force,
                        retry_failed=args.retry_failed)
    
    # Run specific step or full pipeline
    if args.step:
        if args.step == '1':
            pipeline.step_1_fetch_descriptions()
        elif args.step == '2':
            pipeline.step_2_generate_enhanced_descriptions()
        elif args.step == '3':
            pipeline.step_3_generate_embeddings()
        elif args.step == '4':
            pipeline.step_4_create_datasets()
        elif args.step == '5':
            pipeline.step_5_train_and_test_model(epochs=args.epochs)
    else:
        pipeline.run(epochs=args.epochs)


if __name__ == "__main__":
    main()
