#!/usr/bin/env python3
"""
Evaluate answers for BERT-Recall-L compliance.

Since we don't have reference answers, this script provides:
1. Length compliance analysis (to avoid 3x penalty)
2. Relevance heuristic (word overlap with question)
3. Information density scoring
"""

import pandas as pd
import re
from pathlib import Path


def estimate_length_coefficient(answer_chars: int, reference_chars: int = 50) -> float:
    """
    Estimate the length coefficient for BERT-Recall-L.
    
    Args:
        answer_chars: Length of generated answer in characters
        reference_chars: Estimated reference length (default 50 chars)
        
    Returns:
        Length coefficient (0.0 to 1.0)
    """
    if answer_chars <= 1.5 * reference_chars:
        return 1.0
    elif answer_chars < 3 * reference_chars:
        # Linear decrease from 1.0 to 0.0
        return 1.0 - (answer_chars - 1.5 * reference_chars) / (1.5 * reference_chars)
    else:
        return 0.0


def evaluate_answers(
    submission_path: str = "data/submission_test.csv",
    questions_path: str = "data/questions.csv",
    reference_length: int = 50,
) -> dict:
    """
    Evaluate answers for quality and length compliance.
    
    Args:
        submission_path: Path to submission CSV
        questions_path: Path to questions CSV
        reference_length: Assumed reference answer length in chars
        
    Returns:
        Dictionary with evaluation statistics
    """
    # Load data
    answers_df = pd.read_csv(submission_path)
    questions_df = pd.read_csv(questions_path)
    
    # Merge on q_id
    merged = answers_df.merge(questions_df, on="q_id", suffixes=("_answer", "_question"))
    
    # Calculate metrics
    merged["answer_chars"] = merged["answer"].str.len()
    merged["answer_words"] = merged["answer"].str.split().str.len()
    
    # Length coefficient (assuming reference is ~50 chars)
    merged["length_coeff"] = merged["answer_chars"].apply(
        lambda x: estimate_length_coefficient(x, reference_length)
    )
    
    # Check for "Недостаточно информации"
    merged["has_no_info"] = merged["answer"].str.contains(
        "Недостаточно информации", case=False, na=False
    )
    
    # Check for empty answers
    merged["is_empty"] = merged["answer_chars"] == 0
    
    # Calculate word overlap with question
    def word_overlap(row):
        query_words = set(str(row["query"]).lower().split())
        answer_words = set(str(row["answer"]).lower().split())
        if not query_words:
            return 0
        return len(query_words & answer_words) / len(query_words)
    
    merged["word_overlap"] = merged.apply(word_overlap, axis=1)
    
    # Information density (contains actionable words)
    actionable_patterns = [
        r"можно|нужно|должен|обязательно|следует|требуется",
        r"зайдите|перейдите|нажмите|выберите|введите|откройте",
        r"позвоните|обратитесь|напишите|заполните",
        r"будет|может|можно|доступен|предоставляется",
    ]
    
    def has_actionable(text):
        if not text:
            return False
        for pattern in actionable_patterns:
            if re.search(pattern, str(text).lower()):
                return True
        return False
    
    merged["has_actionable"] = merged["answer"].apply(has_actionable)
    
    # Statistics
    stats = {
        "total": len(merged),
        "avg_chars": merged["answer_chars"].mean(),
        "avg_words": merged["answer_words"].mean(),
        "avg_length_coeff": merged["length_coeff"].mean(),
        "no_info_count": merged["has_no_info"].sum(),
        "empty_count": merged["is_empty"].sum(),
        "long_penalty_risk": (merged["length_coeff"] < 1.0).sum(),
        "zero_penalty_risk": (merged["length_coeff"] == 0.0).sum(),
        "avg_word_overlap": merged["word_overlap"].mean(),
        "actionable_count": merged["has_actionable"].sum(),
    }
    
    # Print report
    print("=" * 80)
    print("BERT-RECALL-L METRIC EVALUATION")
    print("=" * 80)
    
    print(f"\n### LENGTH ANALYSIS (reference ~{reference_length} chars) ###")
    print(f"Average answer length: {stats['avg_chars']:.1f} chars")
    print(f"Average word overlap: {stats['avg_word_overlap']:.2%}")
    print(f"Average length coefficient: {stats['avg_length_coeff']:.2%}")
    
    print(f"\n### PENALTY RISK ###")
    print(f"Answers with length penalty (<1.0 coeff): {stats['long_penalty_risk']} ({stats['long_penalty_risk']/len(merged)*100:.1f}%)")
    print(f"Answers with ZERO score (3x+ penalty): {stats['zero_penalty_risk']} ({stats['zero_penalty_risk']/len(merged)*100:.1f}%)")
    
    print(f"\n### QUALITY ###")
    print(f"'Недостаточно информации' count: {stats['no_info_count']}")
    print(f"Empty answers: {stats['empty_count']}")
    print(f"Answers with actionable info: {stats['actionable_count']} ({stats['actionable_count']/len(merged)*100:.1f}%)")
    
    # Length distribution
    print(f"\n### LENGTH DISTRIBUTION ###")
    bins = [0, 50, 100, 150, 200, 250, 300, 400, 500]
    labels = ["0-50", "51-100", "101-150", "151-200", "201-250", "251-300", "301-400", "401+"]
    merged["length_bin"] = pd.cut(merged["answer_chars"], bins=bins, labels=labels)
    print(merged["length_bin"].value_counts().sort_index())
    
    # Show worst offenders
    print(f"\n### LONGEST ANSWERS (potential penalty) ###")
    longest = merged.nlargest(10, "answer_chars")
    for _, row in longest.iterrows():
        print(f"  {row['q_id']}: {row['answer_chars']} chars, coeff={row['length_coeff']:.2f}")
        print(f"    Q: {row['query'][:60]}...")
        print(f"    A: {row['answer'][:80]}...")
    
    return stats


if __name__ == "__main__":
    evaluate_answers()