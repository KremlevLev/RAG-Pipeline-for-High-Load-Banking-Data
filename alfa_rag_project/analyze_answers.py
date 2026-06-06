#!/usr/bin/env python3
"""Analyze submission_test.csv answers for quality and length metrics."""

import pandas as pd
import re
from pathlib import Path


def analyze_answers():
    """Analyze answers for length and quality."""
    # Load data
    answers_df = pd.read_csv("data/submission_test.csv")
    questions_df = pd.read_csv("data/questions.csv")
    
    # Merge on q_id
    merged = answers_df.merge(questions_df, on="q_id", suffixes=("_answer", "_question"))
    
    # Analyze first 50 answers
    first_50 = merged.head(50).copy()
    
    print("=" * 80)
    print("ANALYSIS OF FIRST 50 ANSWERS")
    print("=" * 80)
    
    # Length analysis
    first_50["answer_chars"] = first_50["answer"].str.len()
    first_50["answer_words"] = first_50["answer"].str.split().str.len()
    
    print("\n### LENGTH STATISTICS ###")
    print(f"Min chars: {first_50['answer_chars'].min()}")
    print(f"Max chars: {first_50['answer_chars'].max()}")
    print(f"Mean chars: {first_50['answer_chars'].mean():.1f}")
    print(f"Min words: {first_50['answer_words'].min()}")
    print(f"Max words: {first_50['answer_words'].max()}")
    print(f"Mean words: {first_50['answer_words'].mean():.1f}")
    
    # Check for "Недостаточно информации"
    has_no_info = first_50["answer"].str.contains("Недостаточно информации", case=False, na=False)
    print(f"\n### 'Недостаточно информации' count: {has_no_info.sum()}")
    
    # Check for empty answers
    empty_answers = first_50["answer"].str.len() == 0
    print(f"### Empty answers count: {empty_answers.sum()}")
    
    # Quality analysis - check if answer contains keywords from question
    print("\n### SAMPLE COMPARISONS (QID, Question, Answer) ###")
    for i, row in first_50.head(20).iterrows():
        qid = row["q_id"]
        question = row["query"][:60] + "..." if len(row["query"]) > 60 else row["query"]
        answer = row["answer"][:100] + "..." if len(row["answer"]) > 100 else row["answer"]
        chars = row["answer_chars"]
        words = row["answer_words"]
        
        # Check if answer is too long (potential penalty)
        penalty_risk = "⚠️ LONG" if chars > 150 else ""
        
        print(f"\n{qid}. Q: {question}")
        print(f"   A ({chars} chars, {words} words) {penalty_risk}: {answer}")
    
    # Detailed analysis of specific patterns
    print("\n" + "=" * 80)
    print("DETAILED PATTERN ANALYSIS")
    print("=" * 80)
    
    # Answers that are very short (potential issues)
    short_answers = first_50[first_50["answer_chars"] < 30]
    print(f"\n### Very short answers (<30 chars): {len(short_answers)} ###")
    for i, row in short_answers.iterrows():
        print(f"  {row['q_id']}: {row['answer']}")
    
    # Answers that are very long (potential penalty)
    long_answers = first_50[first_50["answer_chars"] > 200]
    print(f"\n### Long answers (>200 chars) - potential penalty: {len(long_answers)} ###")
    for i, row in long_answers.iterrows():
        print(f"  {row['q_id']} ({row['answer_chars']} chars): {row['answer'][:80]}...")
    
    # Check for specific patterns
    print("\n### ANSWER TYPE DISTRIBUTION ###")
    
    # Count answers containing phone numbers
    has_phone = first_50["answer"].str.contains(r"8\s*\(?\d{3,4}\s*\)?[\d\s\-]{6,}", regex=True, na=False)
    print(f"Contains phone number: {has_phone.sum()}")
    
    # Count answers containing "БИК"
    has_bik = first_50["answer"].str.contains("БИК", case=False, na=False)
    print(f"Contains БИК: {has_bik.sum()}")
    
    # Count answers containing "счёт" or "счет"
    has_account = first_50["answer"].str.contains(r"с[ёе]т[ьч]", case=False, regex=True, na=False)
    print(f"Contains 'счёт/счет': {has_account.sum()}")
    
    # Count answers containing "приложение" or "мобильный"
    has_app = first_50["answer"].str.contains(r"приложение|мобильн", case=False, regex=True, na=False)
    print(f"Contains 'приложение/мобильный': {has_app.sum()}")
    
    # Count answers containing "обратиться" or "техподдерж"
    has_contact = first_50["answer"].str.contains(r"обратиться|техподдерж", case=False, regex=True, na=False)
    print(f"Contains 'обратиться/техподдерж': {has_contact.sum()}")
    
    # Save analysis to file
    first_50[["q_id", "query", "answer", "answer_chars", "answer_words"]].to_csv(
        "data/analysis_first_50.csv", index=False
    )
    print("\n### Analysis saved to data/analysis_first_50.csv ###")
    
    # ── RECOMMENDATIONS ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 80)
    
    # Calculate what the optimal char limit should be
    # Based on the analysis, most good answers are <150 chars
    # For BERT-Recall-L, we want to stay under 1.5x reference length
    # If reference is ~50-100 chars, we should aim for 150-200 max
    
    print(f"\nCurrent MAX_RESPONSE_CHARS: 450")
    print(f"Recommended MAX_RESPONSE_CHARS: 150-200 (to avoid length penalty)")
    print(f"\nCurrent MAX_SENTENCES: 3")
    print(f"Recommended MAX_SENTENCES: 1-2 (for more concise answers)")
    
    # Show distribution
    print("\n### Answer length distribution ###")
    bins = [0, 50, 100, 150, 200, 250, 300, 400, 500]
    labels = ["0-50", "51-100", "101-150", "151-200", "201-250", "251-300", "301-400", "401+"]
    first_50["length_bin"] = pd.cut(first_50["answer_chars"], bins=bins, labels=labels)
    print(first_50["length_bin"].value_counts().sort_index())


if __name__ == "__main__":
    analyze_answers()