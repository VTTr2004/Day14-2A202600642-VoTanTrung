import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        ragas_scores = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        cluster = self._failure_cluster(test_case, ragas_scores, judge_result)
        status = "pass" if judge_result["final_score"] >= 3.5 and ragas_scores["retrieval"]["hit_rate"] >= 1.0 else "fail"

        return {
            "id": test_case.get("id"),
            "test_case": test_case["question"],
            "expected_answer": test_case["expected_answer"],
            "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
            "agent_response": response["answer"],
            "retrieved_ids": response.get("retrieved_ids", []),
            "latency": round(latency, 4),
            "tokens_used": response.get("metadata", {}).get("tokens_used", 0),
            "prompt_tokens": response.get("metadata", {}).get("prompt_tokens", 0),
            "completion_tokens": response.get("metadata", {}).get("completion_tokens", 0),
            "estimated_cost_usd": response.get("metadata", {}).get("estimated_cost_usd", 0.0),
            "model": response.get("metadata", {}).get("model"),
            "ragas": ragas_scores,
            "judge": judge_result,
            "failure_cluster": cluster,
            "status": status,
            "metadata": test_case.get("metadata", {}),
        }

    def _failure_cluster(self, test_case: Dict, ragas_scores: Dict, judge_result: Dict) -> str:
        if judge_result["final_score"] >= 3.5 and ragas_scores["retrieval"]["hit_rate"] >= 1.0:
            return "none"
        if ragas_scores["retrieval"]["hit_rate"] < 1.0:
            return "retrieval_miss"
        if test_case.get("metadata", {}).get("type") in {"prompt-injection", "red-team"}:
            return "safety_policy"
        if ragas_scores["relevancy"] < 0.45:
            return "incomplete_or_hallucinated"
        return "generation_quality"

    async def run_all(self, dataset: List[Dict], batch_size: int = 10) -> List[Dict]:
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            results.extend(await asyncio.gather(*tasks))
        return results
