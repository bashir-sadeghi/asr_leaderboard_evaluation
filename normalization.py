"""Cumulative scoring policies, from the leaderboard rules to strict word comparison.

Reuses the pinned Whisper/Hugging Face normalizer (Apache-2.0; see
leaderboard_normalizer/LICENSE). Its source files remain unchanged.
"""

import re
import unicodedata

from leaderboard_normalizer.normalizer import (
    EnglishTextNormalizer,
    remove_non_speech_annotations,
    remove_symbols_and_diacritics,
)


NORMALIZATION_REVISION = "cumulative-v1"
LEVEL_LABELS = (
    "Leaderboard", "+ hesitations", "+ names", "+ spelling", "+ numbers",
    "+ contractions", "+ word spacing", "+ case", "+ punctuation",
)


class LevelNormalizer:
    """Each level retains all earlier restrictions. Both texts use the same policy."""

    def __init__(self, level=0):
        if level not in range(9):
            raise ValueError("Normalization level must be between 0 and 8.")
        self.level = level
        self.base = EnglishTextNormalizer()
        self.merge_compounds = level < 6

    def __call__(self, text):
        if self.level == 0:
            return self.base(text)  # Exact leaderboard behavior.

        s = text.lower() if self.level < 7 else text
        s = remove_non_speech_annotations(s)
        # Level 1 onward: keep the hesitations removed by the leaderboard.

        if self.level < 5:
            s = re.sub(r"\s+'", "'", s)
            for pattern, replacement in self.base.replacers.items():
                # Bare clock times are a number rule, not a contraction rule.
                if self.level >= 4 and pattern == r"\b(\d{1,2}):00\b":
                    continue
                s = re.sub(pattern, replacement, s)

        if self.level < 8:
            # Numeric punctuation stays under the shared punctuation policy.
            s = re.sub(r"(\d),(\d)", r"\1\2", s)
            s = re.sub(r"\.([^0-9]|$)", r" \1", s)
            keep = ".%$¢€£"
            if self.level >= 5:
                # Keep apostrophes inside contractions, but ignore quotation marks.
                s = unicodedata.normalize("NFKD", s)
                s = re.sub(r"(?<!\w)['’]|['’](?!\w)", " ", s)
                keep += "'’"
        else:
            keep = "".join(c for c in unicodedata.normalize("NFKD", s)
                           if unicodedata.category(c)[0] in "SP")
        # Accent normalization is shared by every level, including level 8.
        s = remove_symbols_and_diacritics(s, keep=keep)

        for pattern, replacement in self.base.compound_words.items():
            # The upstream table mixes spacing rules with spellings like ohhh -> oh.
            threshold = 6 if r"\s" in pattern else 3
            if self.level < threshold:
                s = re.sub(pattern, replacement, s)

        if self.level < 4:
            s = self.base.standardize_numbers(s)
        if self.level < 3:
            s = self.base.standardize_spellings(s)
        if self.level < 2:
            s = self.base.standardize_names(s)
        if self.level < 6:
            s = self.base.standardize_acronyms(s)

        if self.level < 8:
            s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
            s = re.sub(r"([^0-9])%", r"\1 ", s)
        return " ".join(s.split())
