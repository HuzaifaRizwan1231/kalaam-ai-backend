from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List, Optional


class TopicCoverageAnalyzer:
    """Analyzer for computing semantic coverage of a specific topic in a transcript"""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks of words"""
        if not text:
            return []
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            if len(chunk) < 50:  # ignore very tiny chunks
                continue
            chunks.append(" ".join(chunk))
        
        # If the text was too short for any chunks (but not empty), take the whole thing
        if not chunks and words:
            chunks.append(" ".join(words))
            
        return chunks

    def compute_coverage(self, topic: str, transcript: str, threshold: float = 0.5) -> Dict:
        """
        Compute how well the topic is covered in the transcript using semantic similarity.
        """
        if not transcript or not topic:
            return {
                "mean_similarity": 0.0,
                "max_similarity": 0.0,
                "chunks_analyzed": 0,
                "status": "No content to analyze"
            }

        # Chunk transcript
        chunks = self.chunk_text(transcript)
        
        if not chunks:
            return {
                "mean_similarity": 0.0,
                "max_similarity": 0.0,
                "chunks_analyzed": 0,
                "status": "Content too short for analysis"
            }

        # Embed topic and chunks
        topic_embedding = self.model.encode(topic, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)

        # Compute cosine similarities between topic and all chunks
        similarities = util.cos_sim(topic_embedding, chunk_embeddings)[0]
        
        # Convert to numpy for calculations
        similarities = similarities.cpu().numpy()
        
        mean_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        return {
            "mean_similarity": float(mean_similarity),
            "max_similarity": float(max_similarity),
            "topic": topic,
            "chunks_analyzed": len(chunks)
        }
