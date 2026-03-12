"""
RAGAS Evaluation Script — measures RAG pipeline quality against a golden dataset.
Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generation.llm import GroqLLM
from src.generation.prompt_manager import PromptManager
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


def load_golden_dataset(path: str) -> List[dict]:
    """Load the golden Q&A dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_thresholds(config_path: str = None) -> dict:
    """Load evaluation thresholds from settings."""
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "config", "settings.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("evaluation", {}).get("thresholds", {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
        "context_recall": 0.6,
    })


def evaluate_faithfulness(answer: str, context_chunks: List[str]) -> float:
    """
    Measure if claims in the answer are supported by the context.
    Simple heuristic: check overlap between answer sentences and context.

    For production use, integrate RAGAS library for LLM-based evaluation.
    """
    if not answer or not context_chunks:
        return 0.0

    context_text = " ".join(context_chunks).lower()
    answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]

    if not answer_sentences:
        return 0.0

    supported = 0
    for sentence in answer_sentences:
        words = sentence.lower().split()
        if len(words) < 3:
            supported += 1
            continue

        # Check if key words from the sentence appear in context
        key_words = [w for w in words if len(w) > 3]
        if not key_words:
            supported += 1
            continue

        matches = sum(1 for w in key_words if w in context_text)
        if matches / len(key_words) > 0.3:
            supported += 1

    return supported / len(answer_sentences)


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    Measure if the answer addresses the question.
    Simple heuristic: keyword overlap between question and answer.
    """
    if not answer or not question:
        return 0.0

    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())

    # Remove common stop words
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "which", "who", "in", "of", "to",
                  "and", "or", "for", "on", "at", "by", "it", "do", "does"}

    question_keywords = question_words - stop_words
    answer_keywords = answer_words - stop_words

    if not question_keywords:
        return 1.0

    overlap = len(question_keywords & answer_keywords)
    return min(overlap / len(question_keywords), 1.0)


def evaluate_context_precision(question: str, context_chunks: List[str]) -> float:
    """
    Measure if retrieved chunks are relevant to the question.
    Higher precision = less noise in retrieved context.
    """
    if not context_chunks:
        return 0.0

    question_words = set(question.lower().split())
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "which", "who", "in", "of", "to",
                  "and", "or", "for", "on", "at", "by", "it", "do", "does"}
    question_keywords = question_words - stop_words

    if not question_keywords:
        return 1.0

    relevant_chunks = 0
    for chunk in context_chunks:
        chunk_lower = chunk.lower()
        matches = sum(1 for w in question_keywords if w in chunk_lower)
        if matches / len(question_keywords) > 0.2:
            relevant_chunks += 1

    return relevant_chunks / len(context_chunks)


def evaluate_context_recall(
    question: str,
    ground_truth: str,
    context_chunks: List[str],
) -> float:
    """
    Measure if the retriever found all relevant information.
    Compares ground truth answer content against retrieved context.
    """
    if not ground_truth or not context_chunks:
        return 0.0

    gt_sentences = [s.strip() for s in ground_truth.split(".") if s.strip()]
    if not gt_sentences:
        return 0.0

    context_text = " ".join(context_chunks).lower()

    recalled = 0
    for sentence in gt_sentences:
        words = [w for w in sentence.lower().split() if len(w) > 3]
        if not words:
            recalled += 1
            continue

        matches = sum(1 for w in words if w in context_text)
        if matches / len(words) > 0.3:
            recalled += 1

    return recalled / len(gt_sentences)


def run_evaluation(
    pipeline: RAGPipeline,
    dataset: List[dict],
    thresholds: dict,
) -> dict:
    """
    Run full evaluation against the golden dataset.

    Returns evaluation report with per-question and aggregate scores.
    """
    results = []
    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(dataset)} questions")
    print(f"{'='*60}\n")

    for i, entry in enumerate(dataset):
        question = entry["question"]
        ground_truth = entry["ground_truth_answer"]

        print(f"[{i+1}/{len(dataset)}] {question[:60]}...")

        try:
            response = pipeline.query(question)

            # Extract context chunks from citations
            context_chunks = [c.chunk_text for c in response.citations]

            # Calculate metrics
            faithfulness = evaluate_faithfulness(response.answer, context_chunks)
            relevancy = evaluate_answer_relevancy(question, response.answer)
            precision = evaluate_context_precision(question, context_chunks)
            recall = evaluate_context_recall(question, ground_truth, context_chunks)

            result = {
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": response.answer[:500],
                "is_grounded": response.is_grounded,
                "confidence_score": response.confidence_score,
                "metrics": {
                    "faithfulness": round(faithfulness, 4),
                    "answer_relevancy": round(relevancy, 4),
                    "context_precision": round(precision, 4),
                    "context_recall": round(recall, 4),
                },
                "chunks_used": response.chunks_used,
            }

            print(f"  ✓ Faith: {faithfulness:.2f} | Rel: {relevancy:.2f} | "
                  f"Prec: {precision:.2f} | Recall: {recall:.2f}")

        except Exception as e:
            result = {
                "question": question,
                "error": str(e),
                "metrics": {
                    "faithfulness": 0,
                    "answer_relevancy": 0,
                    "context_precision": 0,
                    "context_recall": 0,
                },
            }
            print(f"  ✗ Error: {e}")

        results.append(result)

    # Aggregate scores
    aggregate = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        scores = [r["metrics"][metric] for r in results]
        aggregate[metric] = round(sum(scores) / len(scores), 4) if scores else 0

    # Check thresholds
    passed = True
    threshold_results = {}
    for metric, threshold in thresholds.items():
        score = aggregate.get(metric, 0)
        metric_passed = score >= threshold
        threshold_results[metric] = {
            "score": score,
            "threshold": threshold,
            "passed": metric_passed,
        }
        if not metric_passed:
            passed = False

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(dataset),
        "aggregate_scores": aggregate,
        "threshold_results": threshold_results,
        "overall_passed": passed,
        "per_question_results": results,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument(
        "--dataset",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "golden_dataset.json"
        ),
        help="Path to golden dataset JSON"
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Path for output report"
    )
    args = parser.parse_args()

    # Load dataset and thresholds
    dataset = load_golden_dataset(args.dataset)
    thresholds = load_thresholds()

    # Initialize pipeline components
    vector_store = VectorStore()
    bm25 = BM25Search()

    # Try to load existing BM25 index
    if not bm25.load_index():
        chunks = vector_store.get_all_chunks()
        if chunks:
            bm25.build_index(chunks)

    hybrid = HybridRetriever(vector_store, bm25)
    reranker = Reranker()
    llm = GroqLLM()
    prompt_manager = PromptManager()

    pipeline = RAGPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        llm=llm,
        prompt_manager=prompt_manager,
    )

    # Run evaluation
    report = run_evaluation(pipeline, dataset, thresholds)

    # Save report
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for metric, result in report["threshold_results"].items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  {metric}: {result['score']:.4f} (threshold: {result['threshold']}) [{status}]")
    print(f"\n  Overall: {'✓ PASSED' if report['overall_passed'] else '✗ FAILED'}")
    print(f"\n  Report saved to: {args.output}")

    # Exit with non-zero code if failed (for CI gating)
    sys.exit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
