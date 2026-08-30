# Autonomous Enterprise RAG Platform

An enterprise-grade, agentic Retrieval-Augmented Generation (RAG) system built for context-aware document processing, multi-stage hybrid retrieval, and self-correcting agent orchestration.

## Key Features

* **Agentic Orchestration:** Built with LangGraph / LlamaIndex to support multi-step reasoning, self-reflection, and dynamic tool invocation.
* **Multi-Stage Hybrid Search:** Combines dense vector retrieval (Qdrant / FAISS) with sparse keyword search (BM25), followed by Cross-Encoder re-ranking.
* **Latency & Cost Optimization:** Integrated Redis semantic caching to serve frequent queries instantly.
* **Zero-Trust Security:** Secure tool gateway pattern ensuring strict access control during agent executions.
* **Automated Evaluation & Observability:** Production tracing via LangSmith / Arize Phoenix and automated evaluation pipelines using DeepEval / Ragas.

---

## Architecture Overview

```mermaid
graph TD
    User([User / Client]) -->|API Request| FastAPI[FastAPI Backend]
    
    subgraph Security & Optimization
        FastAPI --> Cache{Redis Semantic Cache}
        Cache -->|Cache Hit| ReturnCache[Return Cached Response]
    end
    
    subgraph Multi-Stage Hybrid Retrieval
        Cache -->|Cache Miss| QueryRewriter[Query Rewriter / Router]
        QueryRewriter -->|Dense Search| VectorDB[(Vector DB: Qdrant / FAISS)]
        QueryRewriter -->|Sparse Search| SparseDB[(BM25 Keyword Search)]
        
        VectorDB -->|Top-K Dense Hits| RRF[Reciprocal Rank Fusion]
        SparseDB -->|Top-K Sparse Hits| RRF
        
        RRF --> ReRanker[Cross-Encoder Re-Ranker]
    end
    
    subgraph Agentic Orchestration Loop
        ReRanker -->|Top-N Ranked Chunks| Agent[LangGraph Agent]
        Agent -->|Tool Call| Tools[Zero-Trust Tool Gateway]
        Tools --> Agent
        Agent -->|Self-Correction / Reflection| EvalCheck{Groundedness Check}
        EvalCheck -->|Failed| Rewrite[Refine Query / Retry] --> QueryRewriter
        EvalCheck -->|Passed| FinalLLM[LLM Generator]
    end
    
    subgraph Observability & Evaluation
        Agent -.->|Traces & Logs| Obs[LangSmith / Arize Phoenix]
        EvalCheck -.->|Metrics| Ragas[DeepEval / Ragas CI Pipeline]
    end
    
    FinalLLM --> ReturnResponse[API Response]
    ReturnResponse --> User