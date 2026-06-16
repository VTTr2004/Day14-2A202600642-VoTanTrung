import json
import os


def validate_lab():
    print("[CHECK] Validating lab submission format...")

    required_files = [
        "reports/summary.json",
        "reports/benchmark_results.json",
        "analysis/failure_analysis.md",
    ]

    missing = []
    for path in required_files:
        if os.path.exists(path):
            print(f"[OK] Found: {path}")
        else:
            print(f"[FAIL] Missing file: {path}")
            missing.append(path)

    if missing:
        print(f"\n[FAIL] Missing {len(missing)} required file(s).")
        return False

    try:
        with open("reports/summary.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[FAIL] reports/summary.json is not valid JSON: {exc}")
        return False

    if "metrics" not in data or "metadata" not in data:
        print("[FAIL] summary.json must contain 'metrics' and 'metadata'.")
        return False

    metrics = data["metrics"]
    metadata = data["metadata"]

    print("\n--- Quick Stats ---")
    print(f"Total cases: {metadata.get('total', 'N/A')}")
    print(f"Average score: {metrics.get('avg_score', 0):.2f}")

    if metrics.get("hit_rate") is not None:
        print(f"[OK] Retrieval metrics found (Hit Rate: {metrics['hit_rate'] * 100:.1f}%)")
    else:
        print("[WARN] Missing retrieval metric: hit_rate")

    if metrics.get("mrr") is not None:
        print(f"[OK] MRR found: {metrics['mrr']:.2f}")
    else:
        print("[WARN] Missing retrieval metric: mrr")

    if metrics.get("agreement_rate") is not None:
        print(f"[OK] Multi-judge metric found (Agreement Rate: {metrics['agreement_rate'] * 100:.1f}%)")
    else:
        print("[WARN] Missing multi-judge metric: agreement_rate")

    if data.get("regression", {}).get("decision"):
        print(f"[OK] Regression release gate found: {data['regression']['decision']}")
    else:
        print("[WARN] Missing regression release gate decision.")

    if metadata.get("version"):
        print("[OK] Agent version metadata found.")

    print("\n[READY] Lab submission is ready for grading.")
    return True


if __name__ == "__main__":
    validate_lab()
