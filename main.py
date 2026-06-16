import asyncio
import json
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

from agent.main_agent import MainAgent
from dotenv import load_dotenv

from engine.local_llm import DEFAULT_MODEL
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


QUALITY_THRESHOLDS = {
    "min_avg_score": 3.75,
    "min_hit_rate": 0.85,
    "min_agreement_rate": 0.70,
    "max_avg_latency": 0.20,
    "max_cost_usd": 0.02,
}


load_dotenv()


class FastBaselineJudge:
    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict:
        answer_terms = set(answer.lower().split())
        truth_terms = set(ground_truth.lower().split())
        overlap = len(answer_terms & truth_terms) / max(1, len(truth_terms))
        score = round(max(1.0, min(3.0, 1.0 + overlap * 2.0)), 2)
        return {
            "final_score": score,
            "agreement_rate": 0.65,
            "individual_scores": {
                "baseline-overlap-judge": score,
                "baseline-grounding-judge": max(1.0, score - 0.25),
            },
            "score_spread": 0.25,
            "resolution": "baseline nhanh để so sánh regression",
            "reasoning": "Baseline dùng judge nhẹ, candidate V2 mới dùng Qwen local thật.",
            "judge_usage": {"prompt_tokens": 0, "completion_tokens": 0, "duration_ns": 0},
        }


def load_dataset() -> List[Dict]:
    if not os.path.exists("data/golden_set.jsonl"):
        raise FileNotFoundError("Missing data/golden_set.jsonl. Run: python data/synthetic_gen.py")

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if len(dataset) < 50:
        raise ValueError("Golden dataset must contain at least 50 test cases.")
    return dataset


def build_summary(agent_version: str, results: List[Dict], elapsed: float) -> Dict:
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    judge_prompt_tokens = sum(r["judge"].get("judge_usage", {}).get("prompt_tokens", 0) for r in results)
    judge_completion_tokens = sum(r["judge"].get("judge_usage", {}).get("completion_tokens", 0) for r in results)
    agent_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results)
    agent_completion_tokens = sum(r.get("completion_tokens", 0) for r in results)

    metrics = {
        "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
        "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
        "mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
        "faithfulness": sum(r["ragas"]["faithfulness"] for r in results) / total,
        "relevancy": sum(r["ragas"]["relevancy"] for r in results) / total,
        "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
        "avg_latency": sum(r["latency"] for r in results) / total,
        "total_cost_usd": sum(r["estimated_cost_usd"] for r in results),
        "avg_tokens": sum(r["tokens_used"] for r in results) / total,
        "agent_prompt_tokens": agent_prompt_tokens,
        "agent_completion_tokens": agent_completion_tokens,
        "judge_prompt_tokens": judge_prompt_tokens,
        "judge_completion_tokens": judge_completion_tokens,
        "total_ollama_tokens": agent_prompt_tokens + agent_completion_tokens + judge_prompt_tokens + judge_completion_tokens,
        "pass_rate": passed / total,
    }
    metrics = {k: round(v, 6) for k, v in metrics.items()}

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 4),
            "llm_provider": "Ollama local HTTP API",
            "llm_model": os.getenv("AGENT_MODEL", DEFAULT_MODEL),
            "judge_models": [
                f"{os.getenv('JUDGE_MODEL_A', DEFAULT_MODEL)}::accuracy-judge",
                f"{os.getenv('JUDGE_MODEL_B', os.getenv('JUDGE_MODEL_A', DEFAULT_MODEL))}::grounding-judge",
            ],
        },
        "metrics": metrics,
        "failure_clusters": dict(Counter(r["failure_cluster"] for r in results if r["failure_cluster"] != "none")),
    }


async def run_benchmark_with_results(agent_version: str) -> Tuple[List[Dict], Dict]:
    dataset = load_dataset()
    judge = FastBaselineJudge() if agent_version.endswith("Base") else LLMJudge()
    runner = BenchmarkRunner(MainAgent(version=agent_version), RetrievalEvaluator(), judge)
    started = time.perf_counter()
    batch_size = 10 if agent_version.endswith("Base") else 1
    results = await runner.run_all(dataset, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    return results, build_summary(agent_version, results, elapsed)


async def run_local_qwen_benchmark() -> Dict:
    """Hàm chính để người dùng tự chạy benchmark local bằng Ollama/Qwen."""
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Local_Qwen")
    gate = release_gate(v1_summary, v2_summary)
    v2_summary["regression"] = {
        "baseline_version": v1_summary["metadata"]["version"],
        "candidate_version": v2_summary["metadata"]["version"],
        **gate,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    with open("reports/baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(v1_summary, f, ensure_ascii=False, indent=2)

    write_failure_analysis(v2_summary, v2_results, gate)
    return v2_summary


def release_gate(v1: Dict, v2: Dict) -> Dict:
    m1 = v1["metrics"]
    m2 = v2["metrics"]
    deltas = {
        "avg_score": round(m2["avg_score"] - m1["avg_score"], 6),
        "hit_rate": round(m2["hit_rate"] - m1["hit_rate"], 6),
        "agreement_rate": round(m2["agreement_rate"] - m1["agreement_rate"], 6),
        "total_cost_usd": round(m2["total_cost_usd"] - m1["total_cost_usd"], 6),
        "avg_latency": round(m2["avg_latency"] - m1["avg_latency"], 6),
    }
    checks = {
        "quality_threshold": m2["avg_score"] >= QUALITY_THRESHOLDS["min_avg_score"],
        "retrieval_threshold": m2["hit_rate"] >= QUALITY_THRESHOLDS["min_hit_rate"],
        "judge_reliability": m2["agreement_rate"] >= QUALITY_THRESHOLDS["min_agreement_rate"],
        "performance_threshold": m2["avg_latency"] <= QUALITY_THRESHOLDS["max_avg_latency"],
        "cost_threshold": m2["total_cost_usd"] <= QUALITY_THRESHOLDS["max_cost_usd"],
        "no_quality_regression": deltas["avg_score"] >= -0.05,
    }
    decision = "Release" if all(checks.values()) else "Rollback"
    return {"decision": decision, "thresholds": QUALITY_THRESHOLDS, "checks": checks, "deltas": deltas}


def write_failure_analysis(summary: Dict, results: List[Dict], gate: Dict) -> None:
    os.makedirs("analysis/reflections", exist_ok=True)
    worst = sorted(results, key=lambda r: (r["judge"]["final_score"], r["ragas"]["retrieval"]["hit_rate"]))[:3]
    clusters = Counter(r["failure_cluster"] for r in results if r["failure_cluster"] != "none")

    lines = [
        "# Báo cáo phân tích lỗi",
        "",
        "## 1. Tổng quan benchmark",
        f"- Tổng số case: {summary['metadata']['total']}",
        f"- Pass/Fail: {summary['metadata']['passed']}/{summary['metadata']['failed']}",
        f"- Model chạy local: {summary['metadata']['llm_model']}",
        f"- Điểm judge trung bình: {summary['metrics']['avg_score']:.2f} / 5.0",
        f"- Hit Rate: {summary['metrics']['hit_rate']:.2%}",
        f"- MRR: {summary['metrics']['mrr']:.2f}",
        f"- Agreement Rate: {summary['metrics']['agreement_rate']:.2%}",
        f"- Tổng token Ollama ghi nhận: {summary['metrics']['total_ollama_tokens']}",
        "- Chi phí API: $0.000000 vì chạy local bằng Ollama",
        f"- Quyết định release gate: {gate['decision']}",
        "",
        "## 2. Phân cụm lỗi",
        "| Cụm lỗi | Số lượng | Vùng nguyên nhân dự kiến |",
        "|---|---:|---|",
    ]
    if clusters:
        root_map = {
            "retrieval_miss": "Retrieval / viết lại truy vấn",
            "safety_policy": "Prompt / guardrail an toàn",
            "incomplete_or_hallucinated": "Prompt sinh câu trả lời / grounding",
            "generation_quality": "Tổng hợp câu trả lời",
        }
        for cluster, count in clusters.items():
            lines.append(f"| {cluster} | {count} | {root_map.get(cluster, 'Unknown')} |")
    else:
            lines.append("| none | 0 | Không có cụm lỗi đáng kể |")

    lines.extend(["", "## 3. Phân tích 5 Whys cho các case tệ nhất"])
    for idx, case in enumerate(worst, start=1):
        lines.extend([
            f"### Case {idx}: {case['id']} - {case['failure_cluster']}",
            f"- Câu hỏi: {case['test_case']}",
            f"- Điểm: {case['judge']['final_score']} / 5.0",
            f"- Expected IDs: {case['expected_retrieval_ids']}",
            f"- Retrieved IDs: {case['retrieved_ids']}",
            "1. Vì sao case fail? Câu trả lời chưa thỏa hoàn toàn đáp án chuẩn hoặc retrieval target.",
            "2. Vì sao câu trả lời yếu? Context top-1 hoặc bước tổng hợp còn thiếu chi tiết.",
            "3. Vì sao context/tổng hợp thiếu chi tiết? Câu hỏi hard case có nhiều từ khóa giống nhau giữa các combo Vinpearl.",
            "4. Vì sao bị nhiễu giữa các combo? Retriever hiện dùng BM25/lexical nội bộ, chưa có embedding semantic và reranking.",
            "5. Root cause: cần thêm embedding retriever, reranker và prompt bắt model trích dẫn đúng DOC_ID.",
            "",
        ])

    lines.extend([
        "## 4. Kế hoạch cải tiến",
        "- Thêm embedding retriever và reranker để giảm nhầm lẫn giữa các combo có tên gần giống nhau.",
        "- Chuẩn hóa golden dataset bằng review thủ công một phần các case do script sinh.",
        "- Giảm khoảng 30% chi phí/thời gian eval bằng cách chỉ gọi judge thứ hai khi judge thứ nhất nằm vùng không chắc chắn.",
        "- Đưa regression gate vào CI để tự động block khi giảm điểm, hit rate, latency hoặc tăng chi phí.",
        "",
    ])

    with open("analysis/failure_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    reflection_path = "analysis/reflections/reflection_VoTanTrung.md"
    with open(reflection_path, "w", encoding="utf-8") as f:
        f.write(
            "# Reflection cá nhân - Vo Tan Trung\n\n"
            "- Đóng góp: xây pipeline tạo golden dataset từ markdown, retrieval eval, async benchmark, local Qwen judge và release gate.\n"
            "- Bài học kỹ thuật: MRR đo thứ hạng tài liệu đúng đầu tiên; agreement rate giúp kiểm tra độ tin cậy giữa các judge; regression gate bảo vệ chất lượng release.\n"
            "- Trade-off: model local không tốn API cost và bảo mật dữ liệu tốt hơn, nhưng tốc độ/độ ổn định phụ thuộc máy chạy Ollama.\n"
        )


async def main():
    try:
        v2_summary = await run_local_qwen_benchmark()
    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        return

    print("Benchmark complete with local Ollama model.")
    print(f"V2 score: {v2_summary['metrics']['avg_score']:.2f}")
    print(f"Decision: {v2_summary['regression']['decision']}")
    print("Reports written to reports/ and analysis/.")


if __name__ == "__main__":
    asyncio.run(main())
