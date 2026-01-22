"""
Intelligent document chunking strategies.
Preserves semantic meaning while respecting token limits.
"""

from typing import List, Dict, Optional
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SmartChunker:
    """
    Chunk documents intelligently for RAG.
    
    Key features:
    - Token-aware chunking (not just character count)
    - Semantic preservation (splits on paragraphs/sentences)
    - Configurable overlap for context continuity
    - Metadata preservation
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "gpt-4"
    ):
        """
        Initialize chunker with token-aware settings.
        
        Args:
            chunk_size: Target tokens per chunk
            chunk_overlap: Tokens to overlap between chunks
            model_name: Model for tokenizer (gpt-4, gpt-3.5-turbo, etc)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer for accurate token counting
        # Critical: different models have different tokenizers
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found, using cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # LangChain splitter with smart separators
        # Order matters: tries to split on larger semantic units first
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.count_tokens,
            separators=[
                "\n\n\n",    # Major section breaks
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence endings
                "! ",        # Exclamations
                "? ",        # Questions
                "; ",        # Semicolons
                ", ",        # Commas
                " ",         # Words
                ""           # Characters (last resort)
            ],
            keep_separator=True
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken (accurate for OpenAI models).
        
        Why not len(text)/4? Because tokenization varies:
        - "hello" = 1 token
        - "artificial intelligence" = 2 tokens
        - "🤖" = multiple tokens
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def chunk_document(
        self, 
        text: str, 
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Split document into overlapping chunks.
        
        Args:
            text: Document text
            metadata: Original document metadata
            
        Returns:
            List of chunk dicts with text, metadata, stats
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []
        
        # Split text using LangChain
        chunks = self.splitter.split_text(text)
        
        # Add metadata to each chunk
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_obj = {
                'text': chunk_text,
                'chunk_id': i,
                'token_count': self.count_tokens(chunk_text),
                'char_count': len(chunk_text),
                'metadata': {
                    **(metadata or {}),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                }
            }
            chunk_objects.append(chunk_obj)
        
        logger.info(
            f"✓ Created {len(chunk_objects)} chunks "
            f"(avg {sum(c['token_count'] for c in chunk_objects) / len(chunk_objects):.0f} tokens/chunk)"
        )
        
        return chunk_objects
    
    def chunk_with_headers(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Advanced: Preserve document structure by detecting headers.
        
        Useful for structured documents (reports, manuals).
        Adds header context to each chunk's metadata.
        """
        # TODO: Implement header detection
        # For now, use standard chunking
        return self.chunk_document(text, metadata)
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about chunks.
        
        Useful for debugging and optimization.
        """
        if not chunks:
            return {}
        
        token_counts = [c['token_count'] for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'total_chars': sum(c['char_count'] for c in chunks)
        }


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict] = None
) -> List[Dict]:
    """Convenience function for quick chunking"""
    chunker = SmartChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.chunk_document(text, metadata)
