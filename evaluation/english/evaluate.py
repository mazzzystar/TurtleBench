import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import simple_system_prompt, system_prompt_with_2shots
from functools import partial
from tqdm import tqdm
from models import Model, call_api

# Only load Correct and Incorrect test cases
def load_test_cases(filename):
    test_cases = []
    failed_lines = []
    with open(filename, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            parts = line.strip().split("	|	")
            if len(parts) == 3 and parts[2] in ["Correct", "Incorrect"]:
                test_cases.append(parts)
            else:
                failed_lines.append((line_number, line.strip()))
    
    print(f"Total {len(test_cases)} test cases loaded")
    
    if failed_lines:
        print("Failed to load the following lines:")
        for line_number, line in failed_lines:
            print(f"Line {line_number}: {line}")
    
    return test_cases

def load_stories(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def starts_with_answer(response, answer):
    return response.strip().lower().startswith(answer)

def evaluate_models(models, test_cases, stories, shot_type):
    results = {model.display_name: {'correct': 0, 'total': 0} for model in models}
    logs = {model.display_name: [] for model in models}
    challenging_cases = []
    all_cases = []
    time_logs = []

    log_folder = f"logs_with_{shot_type}shots"
    os.makedirs(log_folder, exist_ok=True)

    # Find the last tested sample
    last_tested = max([i for i in range(len(test_cases), 0, -1) 
                       if os.path.exists(f"{log_folder}/all_cases_simple_prompt_{i}.json")], default=0)

    if last_tested > 0:
        with open(f"{log_folder}/all_cases_simple_prompt_{last_tested}.json", "r", encoding="utf-8") as f:
            all_cases = json.load(f)
        
        # Update results with previously tested samples
        for case in all_cases:
            for model_name, result in case['results'].items():
                if result is not None:
                    results[model_name]['total'] += 1
                    if result.lower() == case['ground_truth'].lower():
                        results[model_name]['correct'] += 1

    # Start from the next untested sample
    start_index = len(all_cases)

    for i, (user_input, story_title, ground_truth) in enumerate(tqdm(test_cases[start_index:]), start_index + 1):
        story = next((s for s in stories if s["title"] == story_title), None)
        if not story:
            print(f"Story not found: {story_title}")
            continue

        prompt = (system_prompt_with_2shots if shot_type == "2" else simple_system_prompt)\
            .replace("{surface}", story["surface"]).replace("{bottom}", story["bottom"])

        case_results = {}
        time_usage = {}

        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_model = {executor.submit(partial(call_api_with_timeout, timeout=20), 
                                               model, prompt, user_input): model for model in models}
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    response, elapsed_time = future.result()
                    time_usage[model.display_name] = elapsed_time
                    if response:
                        case_results[model.display_name] = process_response(response)
                except Exception as exc:
                    print(f'{model.display_name} generated an exception: {exc}')

        for model in models:
            if model.display_name not in case_results:
                continue
            
            results[model.display_name]['total'] += 1
            if case_results[model.display_name].lower() == ground_truth.lower():
                results[model.display_name]['correct'] += 1
            else:
                print(f"Wrong Answer - Model: {model.display_name}, <{story_title}>, Input: {user_input}, "
                    f"Response: {case_results[model.display_name]}, GT: {ground_truth}")


            logs[model.display_name].append({
                "Input": user_input,
                "Response": case_results[model.display_name],
                "GT": ground_truth,
                "Accuracy": f"{results[model.display_name]['correct']}/{results[model.display_name]['total']} "
                            f"({results[model.display_name]['correct']/results[model.display_name]['total']:.2f})"
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
            save_interim_results(log_folder, i, challenging_cases, all_cases, time_logs)
            print_current_rankings(results, i)

    return results, challenging_cases, all_cases, time_logs

def call_api_with_timeout(model, prompt, user_input, timeout=20):
    start_time = time.time()
    try:
        result = call_api(model, prompt, user_input)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error in call_api for model {model.display_name}: {str(e)}")
        return None, elapsed_time

def process_response(response):
    response = response.strip().lower()
    if starts_with_answer(response, "correct"):
        return "Correct"
    elif starts_with_answer(response, "incorrect"):
        return "Incorrect"
    elif starts_with_answer(response, "unknown"):
        return "Unknown"
    else:
        return "Invalid"

def save_interim_results(log_folder, i, challenging_cases, all_cases, time_logs):
    with open(f"{log_folder}/challenging_cases_simple_prompt_{i}.json", "w", encoding="utf-8") as f:
        json.dump(challenging_cases, f, ensure_ascii=False, indent=2)
    with open(f"{log_folder}/all_cases_simple_prompt_{i}.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    with open(f"{log_folder}/time_logs_{i}.json", "w", encoding="utf-8") as f:
        json.dump(time_logs, f, ensure_ascii=False, indent=2)

def print_current_rankings(results, i):
    print(f"\nCurrent rankings after {i} items:")
    current_results = [(name, res['correct'] / res['total'], res['correct'], res['total']) 
                       for name, res in results.items()]
    current_results.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, accuracy, correct, total) in enumerate(current_results, 1):
        print(f"{rank}. {name}: {accuracy:.2f} ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Run turtle benchmark evaluation")
    parser.add_argument("--shot", choices=["0", "2"], default="0", help="Number of shots (0 or 2)")
    parser.add_argument("--models", nargs="+", type=str, 
                        help="List of models to evaluate. If not specified, default models will be used.")
    args = parser.parse_args()

    if args.models:
        try:
            selected_models = [Model[model_name] for model_name in args.models]
        except KeyError as e:
            print(f"Error: Invalid model name {e}. Available models are: {', '.join([m.name for m in Model])}")
            return
    else:
        selected_models = DEFAULT_MODELS
        print(f"Using default models: {', '.join([m.name for m in DEFAULT_MODELS])}")

    if args.shot == "2":
        print("Evaluating models with 2 shots")
    else:
        print("Evaluating models with 0 shots simple prompt")

    test_cases = load_test_cases("data/cases.list")
    stories = load_stories("data/stories.json")
    results, challenging_cases, all_cases, time_logs = evaluate_models(selected_models, test_cases, stories, args.shot)

    print_current_rankings(results, len(test_cases))
    print(f"Evaluation complete. Logs saved in 'logs_with_{args.shot}shots' directory.")

    log_folder = f"logs_with_{args.shot}shots"
    with open(f"{log_folder}/overall_time_usage.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_total_time": {model.display_name: sum(log['time_usage'].get(model.display_name, 0) for log in time_logs) for model in selected_models},
            "model_call_count": {model.display_name: sum(1 for log in time_logs if model.display_name in log['time_usage']) for model in selected_models},
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    DEFAULT_MODELS = [
        Model.GPT_35_TURBO, 
        Model.GPT_4O_MINI, 
        Model.GPT_4O_20240806,
        Model.GPT_O1_MINI_20240912, 
        Model.GPT_O1_PREVIEW_20240912, 
        Model.CLAUDE_3_HAIKU, 
        Model.CLAUDE_35_SONNET, 
        Model.LLAMA_31_405B,
        Model.DEEPSEEK_V25,
        Model.Qwen2_72B
    ]
    main()