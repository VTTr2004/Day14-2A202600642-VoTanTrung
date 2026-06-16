import re
from typing import Dict, List


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokens(text: str) -> set:
    return set(TOKEN_RE.findall(text.lower()))


class RetrievalEvaluator:
    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def answer_similarity(self, expected_answer: str, answer: str) -> float:
        expected = tokens(expected_answer)
        actual = tokens(answer)
        if not expected:
            return 0.0
        return len(expected & actual) / len(expected)

    async def score(self, case: Dict, response: Dict, top_k: int = 3) -> Dict:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        similarity = self.answer_similarity(case.get("expected_answer", ""), response.get("answer", ""))
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        return {
            "faithfulness": round(min(1.0, 0.35 + 0.65 * similarity), 4),
            "relevancy": round(similarity, 4),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": round(mrr, 4),
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids[:top_k],
            },
        }

    async def evaluate_batch(self, dataset: List[Dict], responses: List[Dict]) -> Dict:
        scores = [await self.score(case, resp) for case, resp in zip(dataset, responses)]
        total = max(1, len(scores))
        return {
            "avg_hit_rate": sum(s["retrieval"]["hit_rate"] for s in scores) / total,
            "avg_mrr": sum(s["retrieval"]["mrr"] for s in scores) / total,
        }
