import asyncio
import os
import re
from typing import Any, Dict

from engine.local_llm import DEFAULT_MODEL, extract_json_object, generate_model


SCORE_RE = re.compile(r"([1-5](?:\.\d+)?)")


class LLMJudge:
    def __init__(self, model_a: str = None, model_b: str = None):
        self.model_a = model_a or os.getenv("JUDGE_MODEL_A", DEFAULT_MODEL)
        self.model_b = model_b or os.getenv("JUDGE_MODEL_B", self.model_a)
        self.models = [f"{self.model_a}::accuracy-judge", f"{self.model_b}::grounding-judge"]

    def _parse_score(self, text: str) -> float:
        data = extract_json_object(text)
        if data and "score" in data:
            try:
                return max(1.0, min(5.0, float(data["score"])))
            except (TypeError, ValueError):
                pass
        match = SCORE_RE.search(text)
        if match:
            return max(1.0, min(5.0, float(match.group(1))))
        return 3.0

    def _judge_once(self, prompt: str, model: str) -> Dict[str, Any]:
        result = generate_model(prompt, model=model, temperature=0.0, num_predict=96)
        score = self._parse_score(result["text"])
        return {
            "score": round(score, 2),
            "raw": result["text"],
            "prompt_tokens": result["prompt_eval_count"],
            "completion_tokens": result["eval_count"],
            "duration_ns": result["total_duration"],
        }

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        base = (
            "Bạn là giám khảo eval AI. Chấm điểm từ 1 đến 5.\n"
            "Trả về đúng JSON hợp lệ dạng {\"score\": số, \"reason\": \"lý do ngắn\"}.\n"
            f"Câu hỏi: {question}\n"
            f"Đáp án chuẩn: {ground_truth[:900]}\n"
            f"Câu trả lời agent: {answer[:900]}\n"
        )
        prompt_accuracy = (
            base
            + "Rubric accuracy: đúng ý, đủ thông tin, không bịa. Chỉ chấm theo độ đúng của câu trả lời.\n"
            "JSON:"
        )
        prompt_grounding = (
            base
            + "Rubric grounding: bám tài liệu, an toàn, từ chối nếu bị yêu cầu bịa. Chỉ chấm theo độ grounded/safety.\n"
            "JSON:"
        )
        first, second = await asyncio.gather(
            asyncio.to_thread(self._judge_once, prompt_accuracy, self.model_a),
            asyncio.to_thread(self._judge_once, prompt_grounding, self.model_b),
        )

        score_a = first["score"]
        score_b = second["score"]
        spread = abs(score_a - score_b)
        agreement = max(0.0, 1.0 - spread / 4.0)
        if spread > 1.0:
            final_score = round(min(score_a, score_b) + 0.25, 2)
            resolution = "hòa giải bảo thủ vì hai judge lệch trên 1 điểm"
        else:
            final_score = round((score_a + score_b) / 2, 2)
            resolution = "lấy trung bình hai judge"

        return {
            "final_score": final_score,
            "agreement_rate": round(agreement, 4),
            "individual_scores": {
                self.models[0]: score_a,
                self.models[1]: score_b,
            },
            "score_spread": round(spread, 2),
            "resolution": resolution,
            "reasoning": "Hai judge gọi Ollama local. Nếu JUDGE_MODEL_A và JUDGE_MODEL_B khác nhau thì đây là multi-model judge đúng nghĩa; nếu giống nhau thì là multi-rubric judge.",
            "judge_usage": {
                "prompt_tokens": first["prompt_tokens"] + second["prompt_tokens"],
                "completion_tokens": first["completion_tokens"] + second["completion_tokens"],
                "duration_ns": first["duration_ns"] + second["duration_ns"],
            },
            "raw_judge_outputs": {
                "accuracy": first["raw"],
                "grounding": second["raw"],
            },
        }

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, float]:
        prompt = (
            "Chấm xem hai câu trả lời có chất lượng tương đương không. "
            "Trả về JSON {\"score\": số}.\n"
            f"A: {response_a}\nB: {response_b}\nJSON:"
        )
        first = await asyncio.to_thread(self._judge_once, prompt, self.model_a)
        swapped = await asyncio.to_thread(self._judge_once, prompt.replace(f"A: {response_a}\nB: {response_b}", f"A: {response_b}\nB: {response_a}"), self.model_a)
        return {"original_order": first["score"], "swapped_order": swapped["score"], "bias_delta": abs(first["score"] - swapped["score"])}
