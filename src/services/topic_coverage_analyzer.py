from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List, Optional


class TopicCoverageAnalyzer:
    """
    Computes the semantic relevance and coverage of a transcript relative to a specified topic.
    Uses 'sentence-transformers' (SBERT) to map text into a high-dimensional vector space 
    (embeddings) where semantic similarity can be calculated via Cosine Distance.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initializes the transformer model. 
        Model: 'all-mpnet-base-v2' is a high-accuracy, general-purpose transformer 
        trained on 1B+ sentence pairs.
        """
        self.model = SentenceTransformer(model_name)

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        Splits a long transcript into overlapping windows of words.
        - chunk_size: Number of words per analysis window.
        - overlap: Replaced words between windows to preserve contextual meaning at cut points.
        
        This prevents context loss and handles transcripts that exceed the transformer's 
        internal token limit (usually 512 tokens).
        """
        if not text:
            return []
            
        words = text.split()
        chunks = []
        
        # Sliding window approach
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            # Discard ultra-short tail segments to maintain statistical significance
            if len(chunk) < 50:  
                continue
            chunks.append(" ".join(chunk))
        
        # Fallback for short transcripts
        if not chunks and words:
            chunks.append(" ".join(words))
            
        return chunks

    def compute_coverage(self, topic: str, transcript: str, threshold: float = 0.5) -> Dict:
        """
        Analyzes semantic alignment.
        1. Encodes the specified Topic as a high-dimensional vector.
        2. Encodes all Transcript Chunks as vectors.
        3. Measures the Cosine Similarity between the Topic and each Chunk.
        
        Returns:
            mean_similarity: How 'on-topic' the whole speech was on average.
            max_similarity: The most relevant moment (highest focus).
            chunks_analyzed: Count of windows processed.
        """
        if not transcript or not topic:
            return {
                "mean_similarity": 0.0,
                "max_similarity": 0.0,
                "chunks_analyzed": 0,
                "status": "No content to analyze"
            }

        # Step 1: Pre-process and Segment the transcript
        chunks = self.chunk_text(transcript)
        
        if not chunks:
            return {
                "mean_similarity": 0.0,
                "max_similarity": 0.0,
                "chunks_analyzed": 0,
                "status": "Content too short for analysis"
            }

        # Step 2: Vectorization (Embedding)
        # Converts text into a 768-dimensional numerical vector.
        topic_embedding = self.model.encode(topic, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)

        # Step 3: Semantic Comparison (Cosine Similarity)
        # Result range: -1.0 (opposite) to 1.0 (identical). Usually > 0.0 for natural text.
        similarities = util.cos_sim(topic_embedding, chunk_embeddings)[0]
        
        # Transfer results back from GPU/PyTorch tensor to CPU NumPy
        similarities = similarities.cpu().numpy()
        
        # Step 4: Aggregate Statistical Metrics
        mean_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        return {
            "mean_similarity": float(mean_similarity),
            "max_similarity": float(max_similarity),
            "topic": topic,
            "chunks_analyzed": len(chunks)
        }
