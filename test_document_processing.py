"""
Quick test script for document processing.
Run: python test_document_processing.py
"""

from pathlib import Path
from src.document_processing.loaders import DocumentLoader
from src.document_processing.chunkers import SmartChunker

# Create sample text file for testing
sample_dir = Path("data/sample_documents")
sample_dir.mkdir(parents=True, exist_ok=True)

sample_file = sample_dir / "sample.txt"
with open(sample_file, 'w') as f:
    f.write("""
# Enterprise RAG Systems: A Comprehensive Guide

## Introduction

Retrieval-Augmented Generation (RAG) has become the standard approach for building 
production LLM applications. Unlike fine-tuning, RAG allows you to ground LLM 
responses in your specific documents without retraining the model.

## How RAG Works

The RAG pipeline consists of several key steps:

1. Document Processing: Extract text from various formats (PDF, DOCX, etc.)
2. Chunking: Split documents into semantic units
3. Embedding: Convert chunks into vector representations
4. Storage: Store vectors in a vector database
5. Retrieval: Find relevant chunks for user queries
6. Generation: Use LLM to generate answers from retrieved context

## Why RAG Matters

RAG solves the key problems with using LLMs in production:
- Reduces hallucinations by grounding in facts
- Allows quick updates without retraining
- Provides source attribution
- Works with private/proprietary data

## Best Practices

When implementing RAG systems, consider:
- Chunk size optimization (typically 500-1000 tokens)
- Overlap between chunks for context continuity
- Metadata preservation for filtering
- Embedding model selection
- Retrieval strategy (semantic search, hybrid, etc.)

## Conclusion

RAG represents the practical path to production LLM applications. 
By combining retrieval with generation, we get the best of both worlds: 
accurate, up-to-date information with natural language generation.
""")

print("=" * 60)
print("TESTING DOCUMENT PROCESSING")
print("=" * 60)

# Test 1: Load document
print("\n1. Loading document...")
doc = DocumentLoader.load(sample_file)
print(f"   ✓ Loaded: {doc['metadata']['filename']}")
print(f"   ✓ Length: {len(doc['text'])} characters")
print(f"   ✓ Format: {doc['metadata']['format']}")

# Test 2: Chunk document
print("\n2. Chunking document...")
chunker = SmartChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document(doc['text'], doc['metadata'])
print(f"   ✓ Created {len(chunks)} chunks")

# Test 3: Show first chunk
print("\n3. First chunk preview:")
print(f"   Text: {chunks[0]['text'][:200]}...")
print(f"   Tokens: {chunks[0]['token_count']}")
print(f"   Chunk ID: {chunks[0]['chunk_id']}")

# Test 4: Get statistics
print("\n4. Chunk statistics:")
stats = chunker.get_chunk_stats(chunks)
for key, value in stats.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
