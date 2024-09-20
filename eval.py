import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm

from models import call_api

# Load stories


def load_stories(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test_cases(filename, language):
    test_cases = []
    failed_lines = []
    with open(filename, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if language == 'zh':
                parts = line.replace(" ", "").split("\t")
                if len(parts) != 3 or parts[2] not in ["T", "F", "N"]:
                    failed_lines.append((line_number, line))
                    continue
            else:  # English
                parts = line.strip().split("	|	")
                if len(parts) != 3 or parts[2] not in ["Correct", "Incorrect", "Unknown"]:
                    failed_lines.append((line_number, line))
                    continue
            test_cases.append(parts)

    print(f"Total {len(test_cases)} test cases loaded")

    if failed_lines:
        print("Failed to load the following lines:")
        for line_number, line in failed_lines:
            print(f"Line {line_number}: {line}")

    return test_cases


def load_prompt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def starts_with_answer(response, answer):
    return response.strip().lower().startswith(answer.lower())


def evaluate_models(models, test_cases, stories, shot_type, prompt_template, log_folder, language):
    results = {model_name: {'correct': 0, 'total': 0} for model_name in models}
    logs = {model_name: [] for model_name in models}
    challenging_cases = []
    all_cases = []
    time_logs = []

    filename_prefix = f"all_cases_{language}_shot{shot_type}"

    # Find the last tested sample
    last_tested = max([int(fname.split('_')[-1].split('.')[0]) for fname in os.listdir(log_folder)
                       if fname.startswith(filename_prefix) and fname.endswith('.json')], default=0)

    if last_tested > 0:
        with open(f"{log_folder}/{filename_prefix}_{last_tested}.json", "r", encoding="utf-8") as f:
            all_cases = json.load(f)

        # Update results with previously tested samples
        for case in all_cases:
            for model_name, result in case['results'].items():
                if result is not None:
                    results[model_name]['total'] += 1
                    if is_correct(result, case['ground_truth'], language):
                        results[model_name]['correct'] += 1

    # Start from the next untested sample
    start_index = len(all_cases)

    for i, (user_input, story_title, ground_truth) in enumerate(tqdm(test_cases[start_index:]), start_index + 1):
        story = next((s for s in stories if s["title"] == story_title), None)
        if not story:
            print(f"Story not found: {story_title}")
            continue

        prompt = prompt_template.replace(
            "{surface}", story["surface"]).replace("{bottom}", story["bottom"])

        case_results = {}
        time_usage = {}

        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_model = {executor.submit(partial(call_api_with_timeout, timeout=20),
                                               model_name, prompt, user_input): model_name for model_name in models}
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    response, elapsed_time = future.result()
                    time_usage[model_name] = elapsed_time
                    if response:
                        case_results[model_name] = process_response(
                            response, language)
                except Exception as exc:
                    print(f'{model_name} generated an exception: {exc}')

        for model_name in models:
            if model_name not in case_results:
                continue

            results[model_name]['total'] += 1
            if is_correct(case_results[model_name], ground_truth, language):
                results[model_name]['correct'] += 1
            else:
                print(f"Wrong Answer - Model: {model_name}, <{story_title}>, Input: {user_input}, "
                      f"Response: {case_results[model_name]}, GT: {ground_truth}")

            logs[model_name].append({
                "Input": user_input,
                "Response": case_results[model_name],
                "GT": ground_truth,
                "Accuracy": f"{results[model_name]['correct']}/{results[model_name]['total']} "
                            f"({results[model_name]['correct']/results[model_name]['total']:.2f})"
            })

        all_cases.append({
            "input": user_input,
            "story_title": story_title,
            "ground_truth": ground_truth,
            "results": case_results,
            "time_usage": time_usage
        })
        time_logs.append({"sample": i, "time_usage": time_usage})

        if i % 10 == 0 or i == len(test_cases):
            save_interim_results(log_folder, filename_prefix,
                                 i, challenging_cases, all_cases, time_logs)
            print_current_rankings(results, i)

    return results, challenging_cases, all_cases, time_logs


def call_api_with_timeout(model_name, prompt, user_input, timeout=20):
    start_time = time.time()
    try:
        result = call_api(model_name, prompt, user_input)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error in call_api for model {model_name}: {str(e)}")
        return None, elapsed_time


def process_response(response, language):
    response = response.strip().lower()
    if language == 'zh':
        if starts_with_answer(response, "对"):
            return "T"
        elif starts_with_answer(response, "错"):
            return "F"
        elif starts_with_answer(response, "不知道"):
            return "N"
        else:
            return "Invalid"
    else:  # English
        if starts_with_answer(response, "correct"):
            return "Correct"
        elif starts_with_answer(response, "incorrect"):
            return "Incorrect"
        elif starts_with_answer(response, "unknown"):
            return "Unknown"
        else:
            return "Invalid"


def is_correct(model_output, ground_truth, language):
    if language == 'zh':
        if ground_truth == "T":
            return model_output == "T"
        else:
            return model_output in ["F", "N"]
    else:  # English
        if ground_truth == "Correct":
            return model_output == "Correct"
        else:
            return model_output in ["Incorrect", "Unknown"]



def save_interim_results(log_folder, filename_prefix, i, challenging_cases, all_cases, time_logs):
    with open(f"{log_folder}/challenging_cases_{filename_prefix}_{i}.json", "w", encoding="utf-8") as f:
        json.dump(challenging_cases, f, ensure_ascii=False, indent=2)
    with open(f"{log_folder}/{filename_prefix}_{i}.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    with open(f"{log_folder}/time_logs_{filename_prefix}_{i}.json", "w", encoding="utf-8") as f:
        json.dump(time_logs, f, ensure_ascii=False, indent=2)


def print_current_rankings(results, i):
    print(f"\nCurrent rankings after {i} items:")
    current_results = [(name, res['correct'] / res['total'], res['correct'], res['total'])
                       for name, res in results.items()]
    current_results.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, accuracy, correct, total) in enumerate(current_results, 1):
        print(f"{rank}. {name}: {accuracy:.2f} ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(
        description="Run turtle benchmark evaluation")
    parser.add_argument(
        "--shot", choices=["0", "2"], default="0", help="Number of shots (0 or 2)")
    parser.add_argument("--models", nargs="+", type=str,
                        help="List of models to evaluate. If not specified, default models will be used.")
    parser.add_argument("--language", choices=["en", "zh"], default="zh",
                        help="Language to use for evaluation (en or zh)")
    args = parser.parse_args()

    selected_models = args.models

    print(f"Evaluating models with {args.shot} shots")
    print(f"Using language: {args.language}")

    # Set data paths
    data_path = os.path.join("data", args.language)
    test_cases = load_test_cases(os.path.join(
        data_path, "cases.list"), args.language)
    stories = load_stories(os.path.join(data_path, "stories.json"))

    # Load prompt
    prompt_filename = f"{'prompt_2shots' if args.shot == '2' else 'simple_prompt'}_{args.language}.txt"
    prompt_path = os.path.join("prompts", prompt_filename)
    prompt_template = load_prompt(prompt_path)

    # Set log folder
    log_folder = os.path.join("logs", f"{args.language}_with_{args.shot}shots")
    os.makedirs(log_folder, exist_ok=True)

    # Evaluate models
    results, challenging_cases, all_cases, time_logs = evaluate_models(
        selected_models, test_cases, stories, args.shot, prompt_template, log_folder, args.language)

    print_current_rankings(results, len(test_cases))
    print(f"Evaluation complete. Logs saved in '{log_folder}' directory.")

    # Save overall time usage
    with open(os.path.join(log_folder, "overall_time_usage.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_total_time": {model_name: sum(log['time_usage'].get(model_name, 0) for log in time_logs) for model_name in selected_models},
            "model_call_count": {model_name: sum(1 for log in time_logs if model_name in log['time_usage']) for model_name in selected_models},
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
