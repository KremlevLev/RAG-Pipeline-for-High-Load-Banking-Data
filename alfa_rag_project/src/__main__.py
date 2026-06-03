"""
Entry point for `python -m main` command.
Runs Kaggle-optimized pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from kaggle_main import run_pipeline, KAGGLE_MODELS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle RAG Pipeline")

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build index from scratch",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-7b",
        choices=list(KAGGLE_MODELS.keys()),
        help="LLM model (open-source, Hugging Face)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/answer_cache.json"),
        help="Path to answer cache",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable answer validation",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1,
        help="Minimum word overlap for validation",
    )

    args = parser.parse_args()

    run_pipeline(
        build_index=args.build_index,
        llm_model=args.model,
        cache_path=args.cache_path,
        validate_answers=not args.no_validate,
        min_overlap=args.min_overlap,
    )