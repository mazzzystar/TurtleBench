import argparse
import json
import os
import time
from datetime import datetime

from tqdm import tqdm

from models import call_api


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


def evaluate_model(model_name, test_cases, stories, shot_type, prompt_template, log_folder, language, save_interval, time_delay):
    all_cases = []

    info = {
        "model": model_name,
        "language": language,
        "shot_type": shot_type,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    overall = {
        'total_samples': 0,
        'correct': 0,
        'accuracy': 0.0,
        'time_usage': 0.0,
        'total_tokens': 0
    }

    filename_prefix = f"all_cases_{model_name}_{language}_shot{shot_type}"

    existing_files = [fname for fname in os.listdir(log_folder) if fname.startswith(
        filename_prefix) and fname.endswith('.json') and model_name in fname]

    if existing_files:
        last_tested = max([int(fname.split('_')[-1].split('.')[0])
                          for fname in existing_files])
    else:
        last_tested = 0

    if last_tested > 0:
        with open(f"{log_folder}/{filename_prefix}_{last_tested}.json", "r", encoding="utf-8") as f:
            results = json.load(f)

        all_cases = results['cases']

        for case in all_cases:
            result = case['model_judge']
            if result is not None:
                overall['total_samples'] += 1
                if is_correct(result, case['ground_truth'], language):
                    overall['correct'] += 1
            overall['total_tokens'] += case['total_tokens']
            overall['time_usage'] += case['time_usage']

        overall['accuracy'] = round(
            overall['correct'] / overall['total_samples'], 6) if overall['total_samples'] > 0 else 0.0

    start_index = len(all_cases)

    for i, (user_input, story_title, ground_truth) in enumerate(tqdm(test_cases[start_index:]), start_index + 1):
        time.sleep(time_delay)
        story = next((s for s in stories if s["title"] == story_title), None)
        if not story:
            print(f"Story not found: {story_title}")
            continue

        prompt = prompt_template.replace(
            "{surface}", story["surface"]).replace("{bottom}", story["bottom"])

        case_result = None
        time_usage = None

        response, total_tokens, elapsed_time = call_api_with_timeout(
            model_name, prompt, user_input)
        time_usage = elapsed_time
        if response:
            case_result = process_response(response, language)

        if case_result is not None:
            overall['total_samples'] += 1
            if is_correct(case_result, ground_truth, language):
                overall['correct'] += 1
            else:
                print(f"Wrong Answer - Model: {model_name}, <{story_title}>, Input: {user_input}, "
                      f"Response: {case_result}, GT: {ground_truth}")

        all_cases.append({
            'sample': i,
            "story_title": story_title,
            "input": user_input,
            "model_response": response,
            "ground_truth": ground_truth,
            "model_judge": case_result,
            "total_tokens": total_tokens,
            "time_usage": time_usage
        })

        overall['accuracy'] = round(
            overall['correct'] / overall['total_samples'], 6) if overall['total_samples'] > 0 else 0.0
        overall['total_tokens'] += total_tokens
        overall['time_usage'] += time_usage

        results = {
            "info": info,
            "overall": overall,
            "cases": all_cases
        }

        if i % save_interval == 0 or i == len(test_cases):
            save_interim_results(log_folder, filename_prefix, i, results)
            print(f"Model: {model_name}, Total: {overall['total_samples']}, Correct: {overall['correct']}, "
                  f"Accuracy: {overall['accuracy']}")

    return overall, all_cases


def call_api_with_timeout(model_name, prompt, user_input):
    start_time = time.time()
    try:
        result, total_tokens = call_api(model_name, prompt, user_input)
        elapsed_time = time.time() - start_time
        return result, total_tokens, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error in call_api for model {model_name}: {str(e)}")
        return None, None, elapsed_time


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


def is_correct(model_judge, ground_truth, language):
    if language == 'zh':
        if ground_truth == "T":
            return model_judge == "T"
        else:
            return model_judge in ["F", "N"]
    else:
        if ground_truth == "Correct":
            return model_judge == "Correct"
        else:
            return model_judge in ["Incorrect", "Unknown"]


def save_interim_results(log_folder, filename_prefix, i, all_cases):
    with open(f"{log_folder}/{filename_prefix}_{i}.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Run turtle benchmark evaluation")
    parser.add_argument(
        "--shot", choices=["0", "2"], default="0", help="Number of shots (0 or 2)")
    parser.add_argument("--models", nargs="+", type=str,
                        help="List of models to evaluate. If not specified, default models will be used.")
    parser.add_argument("--language", choices=["en", "zh"], default="zh",
                        help="Language to use for evaluation (en or zh)")
    
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Interval to save interim results")
    
    parser.add_argument("--time_delay", type=float, default=3,
                        help="Time delay between API calls")
    
    args = parser.parse_args()

    if args.models is None:
        selected_models = MODEL_NAMES
    else:
        selected_models = args.models

    print(f"Evaluating models with {args.shot} shots")
    print(f"Using language: {args.language}")

    data_path = os.path.join("data", args.language)
    test_cases = load_test_cases(os.path.join(
        data_path, "cases.list"), args.language)
    stories = load_stories(os.path.join(data_path, "stories.json"))

    prompt_filename = f"{'prompt_2shots' if args.shot == '2' else 'simple_prompt'}_{args.language}.txt"
    prompt_path = os.path.join("prompts", prompt_filename)
    prompt_template = load_prompt(prompt_path)

    log_folder = os.path.join("logs", f"{args.language}_with_{args.shot}shots")
    os.makedirs(log_folder, exist_ok=True)

    for model_name in selected_models:
        print(f"Evaluating model: {model_name}")
        overall, all_cases = evaluate_model(
            model_name, test_cases, stories, args.shot, prompt_template, log_folder, args.language, args.save_interval, args.time_delay)
        print(f"Model: {model_name}, Total: {overall['total_samples']}, Correct: {overall['correct']}, "
              f"Accuracy: {overall['accuracy']}")
        print(
            f"Evaluation complete for model {model_name}. Logs saved in '{log_folder}' directory.")


if __name__ == "__main__":
    MODEL_NAMES = [
        'GPT_4o',
        'Claude_3_5_Sonnet',
        'Moonshot_v1_8k',
        # 'GPT_o1_Preview',
        # 'GPT_o1_Mini',
        'Llama_3_1_405B',
        'Llama_3_1_70B',
        'Deepseek_V2_5',
        'Qwen_2_72B'
    ]
    main()
