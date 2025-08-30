DocuQuery is a sophisticated Retrieval-Augmented Generation (RAG) system that transforms documents into actionable knowledge. It enables users to upload multiple PDFs, ask contextual questions, and receive accurate answers with proper source citations.

Features:-
Multi-PDF Processing: Upload and process multiple PDF documents simultaneously

AI-Powered Querying: Ask complex questions and receive precise, contextual answers

Source Citations: All answers include proper citations to original source material

Advanced RAG Architecture: Combines retrieval with generative AI for optimal results

Professional Interface: Clean, modern UI designed for productivity



## 📊 Configuration

### Chunking Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk Size | 1000 characters | Optimal balance between context and embedding quality |
| Chunk Overlap | 100 characters | Ensures context continuity between chunks |
| Text Splitter | RecursiveCharacterTextSplitter | LangChain's recursive text splitter |

### Retriever & Reranker Settings
| Component | Setting | Value |
|-----------|---------|-------|
| Pinecone | Top-K | 10 chunks initially retrieved |
| Pinecone | Similarity Metric | Cosine similarity |
| Cohere | Model | rerank-v3.5 |
| Cohere | Top-N | 3 chunks after reranking |

### Providers Used
| Service | Provider | Purpose |
|---------|----------|---------|
| Embeddings | Jina AI | Text embedding generation |
| Vector Database | Pinecone | Vector storage and similarity search |
| Reranking | Cohere | Contextual relevance ranking |
| LLM | Google Gemini | Answer generation with citations |
| File Processing | pdfplumber | PDF text extraction |


┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│                 │    │                  │    │                  │
│   Frontend      │    │   Flask API      │    │   Vector Store   │
│   (HTML/JS/CSS) │◄──►│   (Python)       │◄──►│   (Pinecone)     │
│                 │    │                  │    │                  │
└─────────────────┘    └─────────┬────────┘    └──────────────────┘
                                 │
                                 │
                     ┌───────────┴───────────┐
                     │                       │
             ┌───────▼───────┐       ┌───────▼───────┐
             │               │       │               │
             │   Embedding   │       │    LLM &      │
             │   Provider    │       │   Reranker    │
             │   (Jina AI)   │       │  (Gemini +    │
             │               │       │   Cohere)     │
             └───────────────┘       └───────────────┘

