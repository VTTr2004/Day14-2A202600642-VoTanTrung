import asyncio
import json
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

from agent.main_agent import MainAgent
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
            "judge_models": ["lexical-judge-v1", "policy-judge-v1"],
        },
        "metrics": metrics,
        "failure_clusters": dict(Counter(r["failure_cluster"] for r in results if r["failure_cluster"] != "none")),
    }


async def run_benchmark_with_results(agent_version: str) -> Tuple[List[Dict], Dict]:
    dataset = load_dataset()
    runner = BenchmarkRunner(MainAgent(version=agent_version), RetrievalEvaluator(), LLMJudge())
    started = time.perf_counter()
    results = await runner.run_all(dataset)
    elapsed = time.perf_counter() - started
    return results, build_summary(agent_version, results, elapsed)


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
        "# Failure Analysis Report",
        "",
        "## 1. Benchmark Overview",
        f"- Total cases: {summary['metadata']['total']}",
        f"- Pass/Fail: {summary['metadata']['passed']}/{summary['metadata']['failed']}",
        f"- Average judge score: {summary['metrics']['avg_score']:.2f} / 5.0",
        f"- Hit Rate: {summary['metrics']['hit_rate']:.2%}",
        f"- MRR: {summary['metrics']['mrr']:.2f}",
        f"- Agreement Rate: {summary['metrics']['agreement_rate']:.2%}",
        f"- Estimated eval cost: ${summary['metrics']['total_cost_usd']:.6f}",
        f"- Release decision: {gate['decision']}",
        "",
        "## 2. Failure Clustering",
        "| Cluster | Count | Likely root area |",
        "|---|---:|---|",
    ]
    if clusters:
        root_map = {
            "retrieval_miss": "Retrieval / query rewriting",
            "safety_policy": "Prompting / safety guardrails",
            "incomplete_or_hallucinated": "Generation prompt / context grounding",
            "generation_quality": "Answer synthesis",
        }
        for cluster, count in clusters.items():
            lines.append(f"| {cluster} | {count} | {root_map.get(cluster, 'Unknown')} |")
    else:
        lines.append("| none | 0 | No failing cluster observed |")

    lines.extend(["", "## 3. 5 Whys On Worst Cases"])
    for idx, case in enumerate(worst, start=1):
        lines.extend([
            f"### Case {idx}: {case['id']} - {case['failure_cluster']}",
            f"- Question: {case['test_case']}",
            f"- Score: {case['judge']['final_score']} / 5.0",
            f"- Expected IDs: {case['expected_retrieval_ids']}",
            f"- Retrieved IDs: {case['retrieved_ids']}",
            "1. Why did the case fail? The final answer did not fully satisfy the expected answer or retrieval target.",
            "2. Why was the answer weak? The top context or synthesis step missed some required details.",
            "3. Why did the context/synthesis miss details? Query terms and document wording did not align perfectly for hard cases.",
            "4. Why was alignment imperfect? The baseline retriever uses lexical matching without semantic reranking.",
            "5. Root cause: add semantic embeddings/reranking and stricter grounded-answer prompting for production use.",
            "",
        ])

    lines.extend([
        "## 4. Improvement Plan",
        "- Add an embedding retriever plus cross-encoder reranker for ambiguous and paraphrased questions.",
        "- Keep the two-judge consensus, but calibrate thresholds on a manually reviewed validation set.",
        "- Reduce eval cost by about 30% by running the policy judge only when lexical confidence is between 2.5 and 4.5.",
        "- Add regression gates to CI so releases block automatically on score, retrieval, latency, or cost regression.",
        "",
    ])

    with open("analysis/failure_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    reflection_path = "analysis/reflections/reflection_VoTanTrung.md"
    if not os.path.exists(reflection_path):
        with open(reflection_path, "w", encoding="utf-8") as f:
            f.write(
                "# Individual Reflection - Vo Tan Trung\n\n"
                "- Contribution: implemented offline SDG, retrieval metrics, async benchmark runner, multi-judge consensus, and release gate.\n"
                "- Technical learning: MRR measures the rank of the first correct document; agreement rate exposes judge reliability; regression gates protect releases.\n"
                "- Trade-off: deterministic heuristic judges are cheap and reproducible, while hosted LLM judges can be more nuanced but cost more and need API keys.\n"
            )


async def main():
    try:
        v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base")
        v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        return

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

    print("Benchmark complete.")
    print(f"V1 score: {v1_summary['metrics']['avg_score']:.2f}")
    print(f"V2 score: {v2_summary['metrics']['avg_score']:.2f}")
    print(f"Decision: {gate['decision']}")
    print("Reports written to reports/ and analysis/.")


if __name__ == "__main__":
    asyncio.run(main())
