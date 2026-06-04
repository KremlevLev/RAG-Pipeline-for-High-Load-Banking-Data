"""
Test submission script for RAG pipeline.
Processes first 200 questions for quick testing.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

from config import (
    INDEX_PATH,
    QUESTIONS_CSV,
    SUBMISSION_CSV,
    WEBSITES_CSV,
)
from chunker import chunk_all_websites
from generator import create_generator
from indexer import build_and_save_index, load_index
from retriever import create_retriever

# ─────────────────────────────────────────────
# Логирование
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_test_submission(
    build_index: bool = False,
    llm_model: str = "qwen2.5:7b",
    num_questions: int = 200,
) -> None:
    """
    Run pipeline on first N questions for testing.
    
    Args:
        build_index: Whether to build index from scratch
        llm_model: LLM model name for generation
        num_questions: Number of questions to process
    """
    # Build or load index
    if build_index or not INDEX_PATH.exists():
        print("Building index...")
        websites_df = pd.read_csv(WEBSITES_CSV)
        websites_data = [
            (row["web_id"], row["text"])
            for _, row in websites_df.iterrows()
        ]
        chunks = chunk_all_websites(websites_data)
        print(f"Created {len(chunks)} chunks")
        indexer = build_and_save_index(chunks)
    else:
        print("Loading existing index...")
        indexer = load_index()
    
    # Create retriever and generator
    retriever = create_retriever(indexer)
    generator = create_generator(model=llm_model)
    
    # Load questions (first N only)
    print(f"Loading first {num_questions} questions...")
    questions_df = pd.read_csv(QUESTIONS_CSV)
    questions_df = questions_df.head(num_questions)
    
    # Process questions
    print("Generating answers...")
    results = []
    
    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
        q_id = row["q_id"]
        query = row["query"]
        
        # Retrieve context
        try:
            context = retriever.get_context(query)
        except Exception as e:
            logger.error("Retriever failed for q_id=%s: %s", q_id, e)
            context = ""
        
        # Generate answer
        try:
            answer = generator.generate(query, context)
        except Exception as e:
            logger.error("Generator failed for q_id=%s: %s", q_id, e)
            answer = "Недостаточно информации"
        
        results.append({
            "q_id": q_id,
            "answer": answer,
        })
    
    # Save results
    print("Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(SUBMISSION_CSV, index=False)
    print(f"Results saved to {SUBMISSION_CSV} ({len(results_df)} rows)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG Pipeline")
    
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build index from scratch",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="LLM model name",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=200,
        help="Number of questions to process (default: 200)",
    )
    
    args = parser.parse_args()
    
    run_test_submission(
        build_index=args.build_index,
        llm_model=args.model,
        num_questions=args.num_questions,
    )