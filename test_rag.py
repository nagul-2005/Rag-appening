from src.graph import run_rag_pipeline

queries = [
    "How do multi-agent collaboration frameworks function?",
    "What are the categories of Agentic AI systems?",
    "How does human-in-the-loop governance improve AI agent safety?",
]

for question in queries:
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("=" * 70)
    res = run_rag_pipeline(question)
    print(f"Confidence: {res['confidence_score']} | Chunks: {len(res['retrieved_chunks'])}")
    print("\nAnswer:")
    print(res["answer"][:600])
    print("..." if len(res["answer"]) > 600 else "")
