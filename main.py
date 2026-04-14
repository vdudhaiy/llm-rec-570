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
from sentence_transformers import SentenceTransformer

# Import configuration
from config import (
    MOVIES_DESC_FILE, MOVIES_FILE, RATINGS_FILE,
    BASIC_DESC_FILE, REC_DRIVEN_DESC_FILE,
    MOVIES_ENHANCED_DESC_BASIC, 
    MOVIES_ENHANCED_DESC_REC_DRIVEN, MOVIES_ENHANCED_DESC_COMBINED,
    EMBEDDINGS_FILE, EMBEDDINGS_BASIC, EMBEDDINGS_REC_DRIVEN, EMBEDDINGS_COMBINED,
    DESCRIPTIONS_FILE_MAP, EMBEDDINGS_FILE_MAP,
    EMBEDDING_MODEL, EMBEDDING_DIM, MODEL_TYPE, RANDOM_SEED,
    VERBOSE, DEVICE_TYPE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline controller"""
    
    def __init__(self, model_type=None):
        """
        Initialize pipeline with configuration.
        
        Args:
            model_type: Override config model type (A, B, C, or D)
        """
        self.model_type = model_type or MODEL_TYPE
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        
        logger.info("=" * 70)
        logger.info("LLM-Rec Pipeline Initialized")
        logger.info("=" * 70)
        logger.info(f"Model Type: {self.model_type}")
        logger.info("Pipeline will auto-skip completed steps based on output files")
    
    def step_1_fetch_descriptions(self):
        """Step 1: Fetch movie descriptions from TMDB"""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Fetching Movie Descriptions from TMDB")
        logger.info("=" * 70)
        
        if os.path.exists(MOVIES_DESC_FILE):
            logger.info(f"✓ Movies description file already exists: {MOVIES_DESC_FILE}")
            logger.info("Skipping TMDB fetch")
            return
        
        try:
            logger.info("Starting TMDB description fetch...")
            import fetching_descriptions
            # The main block in fetching_descriptions.py will handle the logic
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
        
        # Check if all enhanced description files already exist
        basic_exists = os.path.exists(BASIC_DESC_FILE)
        rec_exists = os.path.exists(REC_DRIVEN_DESC_FILE)
        combined_exists = os.path.exists(MOVIES_ENHANCED_DESC_COMBINED)
        
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
                generateBasicDescriptions(movies, output_file=BASIC_DESC_FILE)
                logger.info(f"✓ Basic descriptions saved to {BASIC_DESC_FILE}")
            else:
                logger.info(f"✓ Basic descriptions already exist: {BASIC_DESC_FILE}")
            
            # Generate recommendation-driven descriptions
            if not rec_exists:
                logger.info("Generating recommendation-driven descriptions...")
                generateRecDrivenDescriptions(movies, output_file=REC_DRIVEN_DESC_FILE)
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
                    output_file=MOVIES_ENHANCED_DESC_COMBINED
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
        embeddings_exist = {
            'A': os.path.exists(EMBEDDINGS_FILE),
            'B': os.path.exists(EMBEDDINGS_BASIC),
            'C': os.path.exists(EMBEDDINGS_REC_DRIVEN),
            'D': os.path.exists(EMBEDDINGS_COMBINED),
        }
        
        # Skip if all required embeddings exist
        all_exist = all(embeddings_exist.values())
        if all_exist:
            logger.info("✓ All embedding files already exist")
            for model_type, exists in embeddings_exist.items():
                if model_type == 'A':
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_FILE}")
                elif model_type == 'B':
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_BASIC}")
                elif model_type == 'C':
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_REC_DRIVEN}")
                else:
                    logger.info(f"  Type {model_type}: {EMBEDDINGS_COMBINED}")
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
                    basic_descriptions = self._read_descriptions_file(BASIC_DESC_FILE)
                    if len(basic_descriptions) == len(movies):
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
                        logger.warning(f"Basic descriptions count mismatch: {len(basic_descriptions)} vs {len(movies)}")
                else:
                    logger.warning(f"Basic descriptions file not found: {BASIC_DESC_FILE}")
            else:
                logger.info(f"✓ Type B embeddings already exist: {EMBEDDINGS_BASIC}")
            
            # Type C: Embeddings from recommendation-driven descriptions only
            if not embeddings_exist['C']:
                logger.info(f"\n--- Type C: Recommendation-Driven Descriptions Only ---")
                if os.path.exists(REC_DRIVEN_DESC_FILE):
                    rec_descriptions = self._read_descriptions_file(REC_DRIVEN_DESC_FILE)
                    if len(rec_descriptions) == len(movies):
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
                        logger.warning(f"Rec-driven descriptions count mismatch: {len(rec_descriptions)} vs {len(movies)}")
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
            
            logger.info("\n✓ Embedding generation complete")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
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
    
    def _save_enhanced_descriptions_file(self, movies_df, descriptions, output_file, description_type):
        """
        Save enhanced descriptions to a DAT file.
        
        Args:
            movies_df: DataFrame with movie metadata
            descriptions: List of enhanced descriptions
            output_file: Output file path
            description_type: Type description (for logging)
        """
        if os.path.exists(output_file):
            logger.info(f"Enhanced descriptions file already exists: {output_file}")
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
        if os.path.exists(output_file):
            logger.info(f"Embeddings already exist: {output_file}")
            return
        
        try:
            logger.info(f"Generating embeddings for {len(movies_df)} movies from {description_type}...")
            
            descriptions = movies_df[desc_column].tolist()
            embeddings = {}
            
            for idx, (movie_id, description) in enumerate(zip(movies_df['movieId'], descriptions)):
                if (idx + 1) % 500 == 0:
                    logger.info(f"  Progress: {idx + 1}/{len(movies_df)} movies")
                
                # Combine movie info with description for context
                title = movies_df.iloc[idx].get('title', '')
                year = movies_df.iloc[idx].get('year', '')
                genres = movies_df.iloc[idx].get('genres', '')
                
                movie_text = (
                    f"Movie title: {title} ({year}). "
                    f"Genres: {genres}. "
                    f"Description: {description}"
                ).strip()
                
                embedding = self.encoder.encode(movie_text).tolist()
                embeddings[int(movie_id)] = embedding
            
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
    
    def step_5_train_and_test_model(self):
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
        }
        
        required_embeddings = embeddings_file_map[self.model_type]
        
        if not os.path.exists(required_embeddings):
            logger.warning(f"Required embeddings file not found: {required_embeddings}")
            logger.warning("Running Step 3 (Embedding Generation) first...")
            self.step_3_generate_embeddings()
        
        if not os.path.exists(RATINGS_FILE) or not os.path.exists(MOVIES_DESC_FILE):
            logger.warning("Dataset files missing, running Step 4 first...")
            self.step_4_create_datasets()
        
        try:
            logger.info(f"Starting model training with type {self.model_type}...")
            import training
            # The training.py main block will handle the logic and save results
            logger.info("✓ Model training and evaluation complete")
            logger.info(f"✓ Test results saved to test_results.json")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def run(self):
        """Run the complete pipeline"""
        try:
            logger.info("Starting LLM-Rec Pipeline")
            
            self.step_1_fetch_descriptions()
            self.step_2_generate_enhanced_descriptions()
            self.step_3_generate_embeddings()
            self.step_4_create_datasets()
            self.step_5_train_and_test_model()
            
            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETE ✓")
            logger.info("=" * 70)
            logger.info("All steps completed successfully!")
            logger.info("Check test_results.json for evaluation metrics")
            
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
        choices=['A', 'B', 'C', 'D'],
        help='Model type to train: A=SimpleCF, B=LLMRec+Basic, C=LLMRec+RecDriven, D=LLMRec+Combined'
    )
    
    parser.add_argument(
        '--step',
        choices=['1', '2', '3', '4', '5'],
        help='Run only a specific pipeline step'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Pipeline(model_type=args.model_type)
    
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
            pipeline.step_5_train_and_test_model()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
