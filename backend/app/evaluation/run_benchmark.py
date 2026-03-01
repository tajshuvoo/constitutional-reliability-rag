import json
import time
from pathlib import Path
from collections import Counter

from backend.app.services.rag_pipeline import ask

DATASET_PATH = Path("backend/data/eval_dataset.json")

# 🔥 Throttle control
SLEEP_SECONDS = 5  # Adjust to 1–5 seconds depending on cost / rate limits


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark():
    dataset = load_dataset()

    total = len(dataset)
    reliable_count = 0
    correct_refusal_count = 0
    retrieval_hit_count = 0
    correction_trigger_count = 0
    correction_success_count = 0

    failure_reasons = Counter()

    print("\n===== RUNNING BENCHMARK =====\n")

    for idx, item in enumerate(dataset, start=1):

        question = item["question"]
        expected_article = item["expected_article"]
        expected_not_found = item["expected_not_found"]

        print(f"\n[{idx}/{total}] Question: {question}")

        start_time = time.time()

        state = ask(question)

        elapsed = time.time() - start_time
        print(f"   → Completed in {elapsed:.2f}s")

        # --- Reliability ---
        if state.reliability_flag:
            reliable_count += 1

        # --- Refusal accuracy ---
        if expected_not_found:
            if state.final_answer.strip().lower() == \
               "the answer is not found in the retrieved constitutional articles.":
                correct_refusal_count += 1

        # --- Retrieval recall ---
        retrieved_articles = {
            str(a.section_no_en) for a in state.retrieved_articles
        }

        if expected_article and expected_article in retrieved_articles:
            retrieval_hit_count += 1

        # --- Correction metrics ---
        if state.correction_triggered:
            correction_trigger_count += 1

        if state.correction_triggered and state.reliability_flag:
            correction_success_count += 1

        if not state.reliability_flag:
            failure_reasons[state.debug_info.get("faithfulness_reason") or
                            state.debug_info.get("citation_reason")] += 1

        # 🔥 Sleep between calls to avoid hammering HF API
        if idx < total:
            print(f"   → Sleeping {SLEEP_SECONDS}s to avoid API burst...")
            time.sleep(SLEEP_SECONDS)

    print("\n===== BENCHMARK RESULTS =====\n")

    print(f"Total Questions: {total}")
    print(f"Reliability Rate: {reliable_count / total:.2%}")
    print(f"Retrieval Recall Rate: {retrieval_hit_count / total:.2%}")
    print(f"Correct Refusal Accuracy: {correct_refusal_count / total:.2%}")
    print(f"Correction Trigger Rate: {correction_trigger_count / total:.2%}")

    if correction_trigger_count > 0:
        print(f"Correction Success Rate: "
              f"{correction_success_count / correction_trigger_count:.2%}")

    print("\nFailure Breakdown:")
    for reason, count in failure_reasons.items():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    run_benchmark()