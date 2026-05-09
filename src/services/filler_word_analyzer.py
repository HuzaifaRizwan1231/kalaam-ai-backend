import nltk
from nltk import word_tokenize, pos_tag, ngrams
from typing import List, Dict

# -----------------------------
# Static NLTK Resource Management
# -----------------------------
# These resources are required for tokenization (punkt) and Part-of-Speech tagging (averaged_perceptron_tagger).
# We ensure they are present at the start of the service lifecycle.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download("punkt_tab", quiet=True)
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)


# -----------------------------
# Filler Word Lexicon
# -----------------------------
# Single word discourse markers often used as crutches.
SINGLE_WORD_FILLERS = {
    "uh", "um", "erm", "er", "ah", "eh", "hmm", "mm",
    "like", "so", "well", "okay", "ok", "right", "actually",
    "basically", "literally", "honestly", "seriously",
    "essentially", "obviously", "clearly", "anyways", "anyway",
    "alright", "oh", "look", "listen", "just"
}

# Common phrases used to stall for time or filler space.
MULTI_WORD_FILLERS = {
    "you know", "i mean", "sort of", "kind of", "you see",
    "the thing is", "or something", "i guess", "i suppose",
    "i think", "i feel like", "as it were", "you get me",
    "and stuff", "or whatever", "like i said",
    "know what i mean", "you know what i'm saying",
    "basically speaking", "just saying"
}


class FillerWordAnalyzer:
    """
    Identifies and quantifies the use of filler words in a speech transcript.
    Uses hybrid logic:
    1. Lexicon lookup for unambiguous fillers (um, uh).
    2. POS Tagging for ambiguous terms (like, so, well) to filter semantic uses.
    3. N-gram analysis for multi-word phrases (you know, i mean).
    """
    
    @staticmethod
    def is_filler(word: str, tag: str, prev_tag: str = None, next_tag: str = None) -> bool:
        """
        Ambiguity Resolution Engine.
        Determines if a candidate word is a filler or a meaningful part of the sentence.
        
        Examples:
        - "I *like* apples." (VERB) → NOT FILLER
        - "He was, *like*, really tall." (ADVERB/MARKER) → FILLER
        """
        lw = word.lower()

        # Unambiguous vocalizations: Almost always fillers.
        if lw in {"uh", "um", "erm", "er", "ah", "eh", "hmm", "mm", "oh"}:
            return True

        # Resolve 'like'
        if lw == "like":
            # If used as a verb (VBZ, VBP, etc.), it's semantic.
            if tag.startswith("VB"):
                return False
            # As a preposition or adverb (IN, RB), it's frequently a filler marker.
            return True

        # Resolve 'so'
        if lw == "so":
            # If used as a conjunction (CC, IN) linking two clauses, it's semantic.
            if tag in {"CC", "IN"}:
                return False
            return True

        # Resolve 'well'
        if lw == "well":
            # If used as an adverb (RB) modifying a following verb, it's a descriptor.
            if tag == "RB" and (next_tag and next_tag.startswith("VB")):
                return False
            return True

        # Resolve 'right'
        if lw == "right":
            # If used as an adjective (JJ) or noun (NN) meaning correctness/direction, it's semantic.
            if tag in {"JJ", "NN"}:
                return False
            return True

        # Default: Lexicon check for words in the primary filler set.
        if lw in SINGLE_WORD_FILLERS:
            return True

        return False

    @staticmethod
    def identify_fillers(text: str) -> Dict:
        """
        Full analysis pass on a transcript text.
        1. Tokenizes and tags parts of speech.
        2. Scans for 1-word fillers with ambiguity resolution.
        3. Scans for 2/3/4-word filler phrases.
        4. Calculates density percentages.
        """
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)

        fillers_found = []

        # Step A: 1-word Filler Scan
        for i, (word, tag) in enumerate(tags):
            prev_tag = tags[i-1][1] if i > 0 else None
            next_tag = tags[i+1][1] if i < len(tags)-1 else None

            if FillerWordAnalyzer.is_filler(word, tag, prev_tag, next_tag):
                fillers_found.append(word)

        # Step B: Multi-word Phrase Scan (sliding window N-grams)
        # Higher N captures longer colloquialisms like 'you know what i mean'.
        lowered_tokens = [t.lower() for t in tokens]
        for n in [2, 3, 4]:
            for gram in ngrams(lowered_tokens, n):
                phrase = " ".join(gram)
                if phrase in MULTI_WORD_FILLERS:
                    fillers_found.append(phrase)

        # Aggregation of counts and unique instances
        filler_counts = {}
        for filler in fillers_found:
            filler_counts[filler] = filler_counts.get(filler, 0) + 1

        total_words = len(tokens)
        total_fillers = len(fillers_found)
        # Density (Percentage): High density (>4-5%) usually indicates nervousness.
        filler_percentage = (total_fillers / total_words * 100) if total_words > 0 else 0

        return {
            "fillers": fillers_found,
            "filler_counts": filler_counts,
            "total_fillers": total_fillers,
            "total_words": total_words,
            "filler_percentage": round(filler_percentage, 2)
        }
