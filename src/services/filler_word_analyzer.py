import nltk
from nltk import word_tokenize, pos_tag, ngrams
from typing import List, Dict

# Download required NLTK resources (only once)
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
# Extensive Filler Word List
# -----------------------------
SINGLE_WORD_FILLERS = {
    "uh", "um", "erm", "er", "ah", "eh", "hmm", "mm",
    "like", "so", "well", "okay", "ok", "right", "actually",
    "basically", "literally", "honestly", "seriously",
    "essentially", "obviously", "clearly", "anyways", "anyway",
    "alright", "oh", "look", "listen", "just"
}

MULTI_WORD_FILLERS = {
    "you know", "i mean", "sort of", "kind of", "you see",
    "the thing is", "or something", "i guess", "i suppose",
    "i think", "i feel like", "as it were", "you get me",
    "and stuff", "or whatever", "like i said",
    "know what i mean", "you know what i'm saying",
    "basically speaking", "just saying"
}


class FillerWordAnalyzer:
    """Service for detecting filler words in transcribed text"""
    
    @staticmethod
    def is_filler(word: str, tag: str, prev_tag: str = None, next_tag: str = None) -> bool:
        """
        Rule-based filter for ambiguous fillers like 'like', 'so', 'well'.
        Returns True if the word is likely being used as a filler.
        """
        lw = word.lower()

        # Direct fillers (safe to always mark)
        if lw in {"uh", "um", "erm", "er", "ah", "eh", "hmm", "mm", "oh"}:
            return True

        # Handle "like"
        if lw == "like":
            # If used as verb (VB*), it's semantic → not filler
            if tag.startswith("VB"):
                return False
            # Otherwise (adverb, discourse marker, IN), likely filler
            return True

        # Handle "so"
        if lw == "so":
            # If tagged as coordinating/subordinating conjunction (CC, IN), semantic
            if tag in {"CC", "IN"}:
                return False
            return True

        # Handle "well"
        if lw == "well":
            # If it's an adverb (RB) modifying a verb → semantic
            if tag == "RB" and (next_tag and next_tag.startswith("VB")):
                return False
            return True

        # Handle "right"
        if lw == "right":
            # If adjective (JJ) or used in math (NN), semantic
            if tag in {"JJ", "NN"}:
                return False
            return True

        # Default: if in filler list, treat as filler
        if lw in SINGLE_WORD_FILLERS:
            return True

        return False

    @staticmethod
    def identify_fillers(text: str) -> Dict:
        """
        Identify filler words in text and return detailed analysis
        Returns: dict with fillers list, counts, and percentage
        """
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)

        fillers_found = []

        # Check single-word fillers
        for i, (word, tag) in enumerate(tags):
            prev_tag = tags[i-1][1] if i > 0 else None
            next_tag = tags[i+1][1] if i < len(tags)-1 else None

            if FillerWordAnalyzer.is_filler(word, tag, prev_tag, next_tag):
                fillers_found.append(word)

        # Check multi-word fillers using n-grams
        lowered_tokens = [t.lower() for t in tokens]
        for n in [2, 3, 4]:
            for gram in ngrams(lowered_tokens, n):
                phrase = " ".join(gram)
                if phrase in MULTI_WORD_FILLERS:
                    fillers_found.append(phrase)

        # Count frequency of each filler
        filler_counts = {}
        for filler in fillers_found:
            filler_counts[filler] = filler_counts.get(filler, 0) + 1

        total_words = len(tokens)
        total_fillers = len(fillers_found)
        filler_percentage = (total_fillers / total_words * 100) if total_words > 0 else 0

        return {
            "fillers": fillers_found,
            "filler_counts": filler_counts,
            "total_fillers": total_fillers,
            "total_words": total_words,
            "filler_percentage": round(filler_percentage, 2)
        }
