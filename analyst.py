import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

from eval import is_correct

STATS_DIR = "./stats"
OUTPUT_DIR = "./outputs"


def load_latest_logs(model_name, language, shot_type):
    """Load the latest cases."""
    filename_prefix = f"all_cases_{model_name}_{language}_shot{shot_type}"

    # Construct log folder path
    log_folder = os.path.join("./logs", f"{language}_with_{shot_type}shots")

    # Find all log files for the model
    log_files = glob.glob(os.path.join(
        log_folder, f"{filename_prefix}_*.json"))

    if not log_files:
        print(f"No log files found for model {model_name} in {log_folder}.")
        return None

    try:
        # Sort log files by the numerical part at the end of the filename
        log_files.sort(key=lambda x: int(
            x.split('_')[-1].split('.')[0]), reverse=True)
    except (ValueError, IndexError) as e:
        print(f"Error sorting log files: {e}")
        return None

    # Load the latest log file (the first one after sorting)
    latest_log_file = log_files[0]

    try:
        with open(latest_log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading file {latest_log_file}: {e}")
        return None

    return data


def calculate_model_metrics(model_name, language, shot_type):
    """Calculate metrics for the model."""
    # Load the cases from the latest log file
    logs = load_latest_logs(model_name, language, shot_type)

    info = logs.get("info", {})
    overall = logs.get("overall", {})
    cases = logs.get("cases", [])

    # Initialize counters
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Calculate metrics
    total_samples = len(cases)
    correct = 0

    story_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for case in cases:
        correct_flag = is_correct(case.get("model_judge"),
                                  case.get("ground_truth"), language)
        case["is_correct"] = correct_flag

        # Update story stats
        story_title = case.get("story_title")
        story_stats[story_title]['total'] += 1

        if correct_flag:
            correct += 1
            story_stats[story_title]['correct'] += 1

        if language == "en":
            if case.get("ground_truth") == "Correct":
                if correct_flag:
                    tp += 1
                else:
                    fn += 1
            else:
                if correct_flag:
                    tn += 1
                else:
                    fp += 1
        else:
            if case.get("ground_truth") == "T":
                if correct_flag:
                    tp += 1
                else:
                    fn += 1
            else:
                if correct_flag:
                    tn += 1
                else:
                    fp += 1

    # Calculate metrics
    accuracy = correct / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0

    # Story accuracy
    story_accuracy = {}
    for story_title, stats in story_stats.items():
        accuracy = stats['correct'] / \
            stats['total'] if stats['total'] > 0 else 0
        story_accuracy[story_title] = accuracy

    average_story_accuracy = sum(
        story_accuracy.values()) / len(story_accuracy) if story_accuracy else 0

    # Check the overall metrics
    assert overall.get("total_samples") == total_samples
    assert overall.get("correct") == correct

    # update the overall metrics
    overall["precision"] = precision
    overall["recall"] = recall
    overall["f1_score"] = f1_score
    overall["conf_matrix"] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    overall["avg_story_accuracy"] = average_story_accuracy
    overall["story_accuracy"] = story_accuracy

    # Save the updated logs to output folder
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(
        OUTPUT_DIR, f"logs_{model_name}_{language}_shot{shot_type}_len{total_samples}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

    print(
        f"Metrics for {model_name} ({language}, shot{shot_type}) have been calculated and logged to {output_path}")

    return logs, logs["overall"]


def aggregate_model_cases(model_names, language, shot_type):
    """Aggregate cases for all models, languages, and shot types."""
    all_cases = {}

    for model_name in model_names:
        # Load logs for the current model
        logs, _ = calculate_model_metrics(model_name, language, shot_type)

        # Aggregate cases
        for case in logs["cases"]:
            key = case["sample"]
            if key not in all_cases:
                all_cases[key] = {
                    "sample": key,
                    "story_title": case["story_title"],
                    "input": case["input"],
                    "ground_truth": case["ground_truth"],
                    f"{model_name}": case["model_judge"],
                }
            else:
                all_cases[key][f"{model_name}"] = case["model_judge"]

        with open(f"./{STATS_DIR}/all_cases_{language}_shot{shot_type}.json", "w", encoding="utf-8") as f:
            json.dump(all_cases, f, indent=4, ensure_ascii=False)

    return all_cases


def find_cases_with_specific_errors(all_cases, error_model, correct_models, language, shot_type):
    """Find cases where a specific model answered incorrectly and specified models answered correctly."""
    specific_error_cases = []

    for case in all_cases.values():
        error_response = case[error_model]
        correct_responses = [case[model] for model in correct_models]

        # Check if the error_model is incorrect and all correct_models are correct
        if not is_correct(error_response, case["ground_truth"], language) and all(is_correct(resp, case["ground_truth"], language) for resp in correct_responses):
            specific_error_cases.append(case)

    # Save the results to a JSON file
    with open(f"./{STATS_DIR}/specific_error_cases_{language}_shot{shot_type}_{len(correct_models)}.json", "w", encoding="utf-8") as f:
        json.dump(specific_error_cases, f, indent=4, ensure_ascii=False)

    return specific_error_cases


def save_stats(model_names, languages, shot_types):
    """Save the stats for all models, languages, and shot types, sorted by language, shot_type, and accuracy, with empty lines between different language or shot_type."""
    if not os.path.exists(STATS_DIR):
        os.makedirs(STATS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = os.path.join(STATS_DIR, f"stats_{timestamp}.csv")

    stats_data = []

    # Collect metrics for all combinations
    for model_name in model_names:
        for language in languages:
            for shot_type in shot_types:
                _, metrics = calculate_model_metrics(
                    model_name, language, shot_type)
                stats_data.append([model_name, language, shot_type,
                                  metrics["total_samples"], metrics["correct"],
                                  metrics["accuracy"], metrics["avg_story_accuracy"], metrics["precision"],
                                  metrics["recall"], metrics["f1_score"],
                                  metrics["conf_matrix"]["TP"], metrics["conf_matrix"]["FP"],
                                  metrics["conf_matrix"]["TN"], metrics["conf_matrix"]["FN"]])

    # Sort stats by language, shot_type, and accuracy (descending)
    # x[1]: language, x[2]: shot_type, x[5]: accuracy (negative for descending)
    stats_data_sorted = sorted(stats_data, key=lambda x: (x[1], x[2], -x[5]))

    # Write to CSV with empty lines between different language and shot_type
    with open(stats_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["Model", "Language", "Shot Type", "Total Samples", "Correct",
                        "Accuracy", "Avg Story Accuracy", "Precision", "Recall", "F1 Score", "TP", "FP", "TN", "FN"])

        # Track the previous language and shot_type
        prev_language = None
        prev_shot_type = None

        # Write sorted data with empty lines where necessary
        for row in stats_data_sorted:
            current_language = row[1]
            current_shot_type = row[2]

            # Check for changes in language or shot_type
            if prev_language is not None and (current_language != prev_language or current_shot_type != prev_shot_type):
                writer.writerow([])  # Insert an empty line

            # Write the current row
            writer.writerow(row)

            # Update the previous language and shot_type
            prev_language = current_language
            prev_shot_type = current_shot_type

    print(f"Stats saved to {stats_file}")


def main(more_analysis=False):
    model_names = [
        # 'GPT_o1_Preview',
        # 'GPT_o1_Mini',
        'GPT_4o',
        'Claude_3_5_Sonnet',
        'Moonshot_v1_8k',
        'Llama_3_1_405B',
        'Llama_3_1_70B',
        'Deepseek_V2_5',
        'Qwen_2_72B'
    ]
    languages = ["en", "zh"]
    shot_types = [2]
    save_stats(model_names, languages, shot_types)

    if more_analysis:
        for language in languages:
            for shot_type in shot_types:
                all_cases = aggregate_model_cases(
                    model_names, language, shot_type)
                find_cases_with_specific_errors(
                    all_cases, "GPT_o1_Preview",
                    ["GPT_4o", "Claude_3_5_Sonnet", "Moonshot_v1_8k", "Llama_3_1_405B",
                        "Llama_3_1_70B", "Deepseek_V2_5", "Qwen_2_72B"],
                    language, shot_type)

                find_cases_with_specific_errors(
                    all_cases, "GPT_o1_Preview",
                    ["GPT_4o", "Claude_3_5_Sonnet", "Llama_3_1_405B"],
                    language, shot_type)


if __name__ == "__main__":
    main()
