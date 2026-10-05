from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_ragas_llm = None
_ragas_emb = None


def _get_ragas_wrappers():
    global _ragas_llm, _ragas_emb
    if _ragas_llm is None or _ragas_emb is None:
        from config import get_chat_model
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings

        chat_llm = get_chat_model()
        _ragas_llm = LangchainLLMWrapper(chat_llm)
        hf_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        _ragas_emb = LangchainEmbeddingsWrapper(hf_emb)
    return _ragas_llm, _ragas_emb


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    import math
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from config import OPENAI_API_KEY

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        if any(len(q) < 5 for q in questions):
            per_question = [
                EvalResult(
                    question=q, answer=a, contexts=c, ground_truth=gt,
                    faithfulness=0.0, answer_relevancy=0.0,
                    context_precision=0.0, context_recall=0.0
                )
                for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
            ]
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "per_question": per_question,
            }

        eval_kwargs = {"raise_exceptions": False}
        if OPENAI_API_KEY:
            try:
                from ragas.run_config import RunConfig
                ragas_llm, ragas_emb = _get_ragas_wrappers()
                answer_relevancy.strictness = 1
                for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
                    m.llm = ragas_llm
                    if hasattr(m, "embeddings"):
                        m.embeddings = ragas_emb
                eval_kwargs["llm"] = ragas_llm
                eval_kwargs["embeddings"] = ragas_emb
                eval_kwargs["run_config"] = RunConfig(timeout=120, max_retries=10, max_wait=90, max_workers=1)
            except Exception as e:
                print(f"  ⚠️  Failed to configure custom RAGAS wrappers: {e}")

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            **eval_kwargs
        )
        df = result.to_pandas()

        def _clean_val(v):
            if v is None:
                return 0.0
            try:
                fv = float(v)
                return 0.0 if math.isnan(fv) else fv
            except (ValueError, TypeError):
                return 0.0

        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]) if isinstance(row["contexts"], (list, tuple)) else [str(row["contexts"])],
                ground_truth=str(row["ground_truth"]),
                faithfulness=_clean_val(row.get("faithfulness")),
                answer_relevancy=_clean_val(row.get("answer_relevancy")),
                context_precision=_clean_val(row.get("context_precision")),
                context_recall=_clean_val(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": _clean_val(result.get("faithfulness")),
            "answer_relevancy": _clean_val(result.get("answer_relevancy")),
            "context_precision": _clean_val(result.get("context_precision")),
            "context_recall": _clean_val(result.get("context_recall")),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        per_question = [
            EvalResult(
                question=q, answer=a, contexts=c, ground_truth=gt,
                faithfulness=0.0, answer_relevancy=0.0,
                context_precision=0.0, context_recall=0.0
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu (LLM hallucinating)",
            "Thắt chặt system prompt, giảm nhiệt độ (temperature) về 0"
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng (Missing relevant chunks)",
            "Cải thiện lại bước cắt đoạn hoặc bổ sung từ khóa BM25"
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp lên đầu (Too many irrelevant chunks)",
            "Bổ sung tầng Cross-Encoder reranking hoặc lọc theo metadata"
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi (Answer doesn't match question)",
            "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn"
        ),
    }

    scored_items = []
    for r in eval_results:
        metrics_scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics_scores, key=metrics_scores.get)
        avg_score = sum(metrics_scores.values()) / 4.0
        diag, fix = diagnostic_tree.get(
            worst_metric,
            ("Unknown issue", "Investigate manually")
        )
        scored_items.append({
            "question": r.question,
            "expected": r.ground_truth,
            "got": r.answer,
            "worst_metric": worst_metric,
            "score": float(metrics_scores[worst_metric]),
            "avg_score": float(avg_score),
            "diagnosis": diag,
            "suggested_fix": fix,
        })

    # Sort by avg_score ascending
    scored_items.sort(key=lambda x: (x["avg_score"], x["score"]))
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
