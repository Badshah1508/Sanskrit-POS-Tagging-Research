# Dataset Audit Report: Panchatantra & Sanskrit News

**Generated:** 2026-11-26T05:28:55.526512+00:00
**Random Seed:** 42

## 1. Source Domain: `Panchatantra_clean.csv`
- **Raw Tokens:** 43,653 across 5,393 sentences
- **Excluded Incomplete Sentences:** 7 sentences (85 tokens containing 6 `NaN` tags and 1 unresolved `UA` tag)
- **Clean Mapped Tokens:** 43,568 across 5,386 sentences (99.98% raw coverage)
- **Token Artifacts:** 113 tokens contain `/` from upstream comma-attached slash tags (preserved transparently)

## 2. Target Domain: `News_Annotated.csv` (Reused Validated Splits)
- **News Adaptation Pool:** 1,484 sentences, 13,544 tokens (1,103 documents)
- **News Validation:** 370 sentences, 3,431 tokens (276 documents)
- **Locked News Test:** 791 sentences, 7,324 tokens (592 documents)

## 3. Panchatantra Splits (70% / 15% / 15%, Duplicate-Free)
- **Panchatantra Train:** 3,928 sentences, 30,594 tokens
- **Panchatantra Validation:** 763 sentences, 6,550 tokens
- **Panchatantra Held-Out Test:** 695 sentences, 6,424 tokens

## 4. Leakage & Integrity Verification
- Zero sentence-text overlap across Panchatantra Train/Val/Test.
- Zero document or sentence overlap across News Pool/Val/Test.
- 100% token-to-tag length parity and 100% `TARGET_TAGS` validity across all 6 splits.
