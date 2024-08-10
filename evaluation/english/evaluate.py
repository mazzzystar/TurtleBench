import os
import json
import asyncio
import requests
import aiohttp
from prompt import simple_system_prompt, system_prompt_with_2shots
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from anthropic import Anthropic
from together import Together
import concurrent.futures
from functools import partial
import threading
from tqdm import tqdm
import argparse
import time

# Load environment variables
load_dotenv()

MAX_TOKENS = 4

# Define the models and their configurations
models = [
    # {
    #     "name": "Gemini-1.5-Pro",
    #     "config": {
    #         "apiKey": os.getenv("GEMINI_API_KEY"),
    #         "model": "gemini-1.5-pro",
    #         "maxTokens": MAX_TOKENS,
    #         "temperature": 0.0,
    #     },
    #     "type": "gemini"
    # },
    {
        "name": "DEEPSEEK",
        "config": {
            "apiKey": os.getenv("DEEPSEEK_API_KEY"),
            "baseURL": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 0.7,
        },
        "type": "openai"
    },
    {
        "name": "GPT-3.5-Turbo",
        "config": {
            "apiKey": os.getenv("OPENAI_API_KEY"),
            "baseURL": "https://api.openai.com/v1",
            "model": "gpt-3.5-turbo",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 0.7,
        },
        "type": "openai"
    },
    # {
    #     "name": "Kimi-Chat",
    #     "config": {
    #         "apiKey": os.getenv("MOONSHOT_API_KEY"),
    #         "baseURL": "https://api.moonshot.cn/v1",
    #         "model": "moonshot-v1-8k",
    #         "maxTokens": MAX_TOKENS,
    #         "temperature": 0.0,
    #         "top_p": 0.7,
    #     },
    #     "type": "openai"
    # },
    {
        "name": "GPT-4o",
        "config": {
            "apiKey": os.getenv("OPENAI_API_KEY"),
            "baseURL": "https://api.openai.com/v1",
            "model": "gpt-4o-2024-05-13",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 0.7,
        },
        "type": "openai"
    },
    {
        "name": "GPT-4o-mini",
        "config": {
            "apiKey": os.getenv("OPENAI_API_KEY"),
            "baseURL": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 0.7,
        },
        "type": "openai"
    },
    {
        "name": "Llama-3.1-405b",
        "config": {
            "apiKey": os.getenv("TOGETHER_API_KEY"),
            "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 0.7,
            "stop": ["<|eot_id|>"]
        },
        "type": "together"
    },
    {
        "name": "Llama3.1-70b",
        "config": {
            "apiKey": os.getenv("TOGETHER_API_KEY"),
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 0.7,
            "stop": ["<|eot_id|>"]
        },
        "type": "together"
    },
    {
        "name": "Qwen2-72B-Instruct",
        "config": {
            "apiKey": os.getenv("TOGETHER_API_KEY"),
            "model": "Qwen/Qwen2-72B-Instruct",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 0.7,
            "stop": ["<|im_start|>", "<|im_end|>"]
        },
        "type": "together"
    },
    # {
    #     "name": "Yi-34B-Chat",
    #     "config": {
    #         "apiKey": os.getenv("TOGETHER_API_KEY"),
    #         "model": "zero-one-ai/Yi-34B-Chat",
    #         "maxTokens": MAX_TOKENS,
    #         "temperature": 0.0,
    #         "top_p": 0.7,
    #         "stop": ["<|im_start|>", "<|im_end|>"]
    #     },
    #     "type": "together"
    # },
    # {
    #     "name": "Doubao-4k",
    #     "config": {
    #         "apiKey": os.getenv("DOUBAO_API_KEY"),
    #         "baseURL": "https://ark.cn-beijing.volces.com/api/v3",
    #         "model": "ep-20240802142948-6vvc7",  # Replace with the actual endpoint ID if different
    #         "maxTokens": MAX_TOKENS,
    #         "temperature": 0.0,
    #         "top_p": 0.7
    #     },
    #     "type": "openai"
    # },
    {
        "name": "Claude-3.5-Sonnet",
        "config": {
            "apiKey": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-5-sonnet-20240620",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
        },
        "type": "anthropic"
    },
    # {
    #     "name": "Claude-3-Opus",
    #     "config": {
    #         "apiKey": os.getenv("ANTHROPIC_API_KEY"),
    #         "model": "claude-3-opus-20240229",
    #         "maxTokens": MAX_TOKENS,
    #         "temperature": 0.0,
    #     },
    #     "type": "anthropic"
    # },
    {
        "name": "Claude-3-Haiku",
        "config": {
            "apiKey": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
        },
        "type": "anthropic"
    },
    # {
    #     "name": "MiniMax-ABAB6.5s",
    #     "config": {
    #         "groupId": os.getenv("MINIMAX_GROUP_ID"),
    #         "apiKey": os.getenv("MINIMAX_API_KEY"),
    #         "model": "abab6.5s-chat",
    #         "maxTokens": MAX_TOKENS,
    #         "temperature": 0.01, # must be (0, 1]
    #         "top_p": 1
    #     },
    #     "type": "minimax"
    # },
]

# Load stories
with open("data/stories.json", "r", encoding="utf-8") as f:
    stories = json.load(f)

def load_test_cases(filename):
    with open(filename, "r", encoding="utf-8") as f:
        _test_cases = []
        for line in f:
            parts = line.strip().split("	|	")
            if len(parts) != 3:
                print(f"Invalid test case: {line}")
                continue
            if parts[2] not in ["Correct", "Incorrect", "Unknown"]:
                print(f"Skipping line with invalid ground truth: {line}")
                continue
            _test_cases.append(parts)
        
        print("Total", len(_test_cases), "test cases loaded")
        return _test_cases

def starts_with_answer(response, answer):
    return response.strip().lower().startswith(answer)

def call_api(model, prompt, user_input):
    try:
        if model["type"] == "openai":
            if model["name"] == "Doubao-4k":
                client = OpenAI(
                    api_key=model["config"]["apiKey"],
                    base_url=model["config"]["baseURL"]
                )
                
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ]
                
                response = client.chat.completions.create(
                    model=model["config"]["model"],
                    messages=messages,
                    max_tokens=model["config"]["maxTokens"],
                    temperature=model["config"]["temperature"],
                    top_p=model["config"]["top_p"],
                    stream=False
                )
                
                return response.choices[0].message.content
            else:
                url = model["config"]["baseURL"] + "/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {model['config']['apiKey']}"
                }
                data = {
                    "model": model["config"]["model"],
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_input}
                    ],
                    "max_tokens": model["config"]["maxTokens"],
                    "temperature": model["config"]["temperature"],
                }
                
                if "top_p" in model["config"]:
                    data["top_p"] = model["config"]["top_p"]

                response = requests.post(url, headers=headers, json=data)
                if response.status_code != 200:
                    raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                result = response.json()
                return result["choices"][0]["message"]["content"]
        
        elif model["type"] == "together":
            client = Together(api_key=model["config"]["apiKey"])
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
            
            response = client.chat.completions.create(
                model=model["config"]["model"],
                messages=messages,
                max_tokens=model["config"]["maxTokens"],
                temperature=model["config"]["temperature"],
                top_p=model["config"]["top_p"],
                stop=model["config"]["stop"],
                stream=False
            )
            
            return response.choices[0].message.content

        elif model["type"] == "anthropic":
            client = Anthropic(api_key=model["config"]["apiKey"])
            
            message = client.messages.create(
                model=model["config"]["model"],
                max_tokens=model["config"]["maxTokens"],
                temperature=model["config"]["temperature"],
                system=prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_input
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text

        elif model["type"] == "minimax":
            url = f"https://api.minimax.chat/v1/text/chatcompletion_v2?GroupId={model['config']['groupId']}"
            headers = {
                "Authorization": f"Bearer {model['config']['apiKey']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model["config"]["model"],
                "messages": [
                    {
                        "role": "system",
                        "name": "MM智能助理",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                "tools": [],
                "tool_choice": "none",
                "stream": False,
                "max_tokens": model["config"]["maxTokens"],
                "temperature": model["config"]["temperature"],
                "top_p": model["config"]["top_p"]
            }
            
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        elif model["type"] == "gemini":
            import google.generativeai as genai
            
            genai.configure(api_key=model["config"]["apiKey"])
            
            generation_config = {
                "temperature": model["config"]["temperature"],
                "max_output_tokens": model["config"]["maxTokens"],
                "top_p": 0.7,
                # "top_k": 64,
            }
            
            gemini_model = genai.GenerativeModel(
                model_name=model["config"]["model"],
                generation_config=generation_config,
            )
            
            chat_session = gemini_model.start_chat(history=[])
            
            # Combine prompt and user_input
            full_prompt = f"{prompt}\n\nUser: {user_input}\nAssistant:"
            
            response = chat_session.send_message(full_prompt)
            
            return response.text
        
        else:
            raise ValueError(f"Unsupported model type: {model['type']}")
    except Exception as e:
        print(f"Error in call_api for model {model['name']}: {str(e)}")
        return None

def call_api_with_timeout_and_timing(model, prompt, user_input, timeout=20):
    start_time = time.time()
    try:
        result = call_api(model, prompt, user_input)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error in call_api for model {model['name']}: {str(e)}")
        return None, elapsed_time

def evaluate_models(models, test_cases, stories, shot_type):
    results = {model['name']: {'correct': 0, 'total': 0} for model in models}
    logs = {model['name']: [] for model in models}
    challenging_cases = []
    all_cases = []
    time_logs = []

    log_folder = f"logs_with_{shot_type}shots"
    os.makedirs(log_folder, exist_ok=True)

    # Find the last tested sample
    last_tested = 0
    for i in range(len(test_cases), 0, -1):
        if os.path.exists(f"{log_folder}/all_cases_simple_prompt_{i}.json"):
            with open(f"{log_folder}/all_cases_simple_prompt_{i}.json", "r", encoding="utf-8") as f:
                all_cases = json.load(f)
            last_tested = i
            break

    # Update results with previously tested samples
    for case in all_cases:
        for model_name, result in case['results'].items():
            if result is not None:
                results[model_name]['total'] += 1
                if (case['ground_truth'] == "Correct" and result == "Correct") or \
                   ((case['ground_truth'] == "Incorrect" or case['ground_truth'] == "Unknown") and result != "Correct"):
                    results[model_name]['correct'] += 1

    # Start from the next untested sample
    start_index = len(all_cases)

    for i, (user_input, story_title, ground_truth) in enumerate(tqdm(test_cases[start_index:]), start_index + 1):
        try:
            story = next((s for s in stories if s["title"] == story_title), None)
            if not story:
                print(f"Story not found: {story_title}")
                continue

            # Use the appropriate prompt based on shot_type
            if shot_type == "2":
                prompt_template = system_prompt_with_2shots
            else:
                prompt_template = simple_system_prompt

            prompt = prompt_template.replace("{surface}", story["surface"]).replace("{bottom}", story["bottom"])
            gt_map = {"correct": "correct", "incorrect": "incorrect", "unknown": "unknown"}

            case_results = {}
            all_responses_valid = True
            time_usage = {}

            # Use ThreadPoolExecutor for concurrent API calls
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
                future_to_model = {executor.submit(partial(call_api_with_timeout_and_timing, timeout=20), model, prompt, user_input): model for model in models}
                for future in concurrent.futures.as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        response, elapsed_time = future.result()
                        time_usage[model['name']] = elapsed_time
                        if response is None:
                            all_responses_valid = False
                            print(f"Timeout or error for model {model['name']}")
                        else:
                            case_results[model['name']] = response
                    except Exception as exc:
                        print(f'{model["name"]} generated an exception: {exc}')
                        all_responses_valid = False

            # If any model timed out or had an error, skip this entire test case
            if not all_responses_valid:
                print(f"Skipping test case {i} due to timeout or error")
                continue

            # Process all responses
            for model in models:
                if model['name'] not in case_results:
                    continue
                response = case_results[model['name']].strip().lower()

                if starts_with_answer(response, "correct") or starts_with_answer(response, "incorrect") or starts_with_answer(response, "unknown"):
                    results[model['name']]['total'] += 1
                    
                    # Save the actual model output
                    if starts_with_answer(response, "correct"):
                        case_results[model['name']] = "Correct"
                    elif starts_with_answer(response, "incorrect"):
                        case_results[model['name']] = "Incorrect"
                    else:
                        case_results[model['name']] = "Unknown"
                    
                    # Calculate accuracy (merging N and F)
                    if (ground_truth.lower() == "correct" and case_results[model['name']].lower() == "correct") or \
                       ((ground_truth.lower() == "incorrect" or ground_truth.lower() == "unknown") and case_results[model['name']].lower() != "correct"):
                        results[model['name']]['correct'] += 1
                    else:
                        # Print only wrong answers
                        print(f"Wrong Answer - Model: {model['name']}, Input: {user_input}, Response: {response}, GT: {ground_truth.lower()}, Model Output: {case_results[model['name']]}")
                else:
                    # Handle invalid responses
                    case_results[model['name']] = "Invalid"
                    print(f"Invalid Response - Model: {model['name']}, Input: {user_input}, Response: {response}, GT: {ground_truth.lower()}, Model Output: {case_results[model['name']]}")

                log_entry = {
                    "Input": user_input,
                    "Response": response,
                    "GT": ground_truth,
                    "Model_Output": case_results[model['name']],
                    "Accuracy": f"{results[model['name']]['correct']}/{results[model['name']]['total']} ({results[model['name']]['correct']/max(results[model['name']]['total'], 1):.2f})"
                }
                logs[model['name']].append(log_entry)

            case = {
                "input": user_input,
                "story_title": story_title,
                "ground_truth": ground_truth,
                "results": case_results,
                "time_usage": time_usage
            }

            all_cases.append(case)
            time_logs.append({"sample": i, "time_usage": time_usage})

            # Print time usage for this sample
            print(f"\nTime usage for sample {i}:")
            for model_name, elapsed_time in sorted(time_usage.items(), key=lambda x: x[1], reverse=True):
                print(f"{model_name}: {elapsed_time:.2f} seconds")

            # Save log and print accuracy ranking every 10 steps
            if i % 10 == 0 or i == len(test_cases):
                print(f"\nCurrent rankings after {i} items:")
                current_results = [(name, res['correct'] / max(res['total'], 1), res['correct'], res['total']) 
                                for name, res in results.items()]
                current_results.sort(key=lambda x: x[1], reverse=True)
                
                for rank, (name, accuracy, correct, total) in enumerate(current_results, 1):
                    print(f"{rank}. {name}: {accuracy:.2f} ({correct}/{total})")

                # Update challenging cases file
                with open(f"{log_folder}/challenging_cases_simple_prompt_{i}.json", "w", encoding="utf-8") as f:
                    json.dump(challenging_cases, f, ensure_ascii=False, indent=2)

                # Update all cases file
                with open(f"{log_folder}/all_cases_simple_prompt_{i}.json", "w", encoding="utf-8") as f:
                    json.dump(all_cases, f, ensure_ascii=False, indent=2)

                # Save time logs
                with open(f"{log_folder}/time_logs_{i}.json", "w", encoding="utf-8") as f:
                    json.dump(time_logs, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing test case {i}: {str(e)}")
            continue

    # Final update to challenging cases file
    final_index = start_index + len(test_cases[start_index:])
    with open(f"{log_folder}/challenging_cases_simple_prompt_{final_index}.json", "w", encoding="utf-8") as f:
        json.dump(challenging_cases, f, ensure_ascii=False, indent=2)

    # Final update to all cases file
    with open(f"{log_folder}/all_cases_simple_prompt_{final_index}.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    return results, challenging_cases, all_cases, time_logs

def save_all_cases(all_cases, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    print(f"All cases have been saved to {output_file}")

def parse_challenging_cases(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        challenging_cases = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for case in challenging_cases:
            user_input = case['input']
            story_title = case['story_title']
            ground_truth = case['ground_truth']
            f.write(f"{user_input}\t{story_title}\t{ground_truth}\n")

    print(f"Parsed challenging cases have been written to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run story understanding evaluation")
    parser.add_argument("--shot", choices=["0", "2"], default="2", help="Number of shots (0 or 2)")
    args = parser.parse_args()

    test_cases = load_test_cases("data/cases.list")
    results, challenging_cases, all_cases, time_logs = evaluate_models(models, test_cases, stories, args.shot)

    final_results = [(name, res['correct'] / max(res['total'], 1), res['correct'], res['total']) 
                     for name, res in results.items()]
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFinal Rankings ({args.shot}-shot):")
    for rank, (name, accuracy, correct, total) in enumerate(final_results, 1):
        print(f"{rank}. {name}: {accuracy:.2f} ({correct}/{total})")

    print(f"Evaluation complete. Logs have been saved in the '{log_folder}' directory.")

    # Analyze and print overall time usage statistics
    model_total_time = {model['name']: 0 for model in models}
    model_call_count = {model['name']: 0 for model in models}

    for log in time_logs:
        for model_name, time_used in log['time_usage'].items():
            model_total_time[model_name] += time_used
            model_call_count[model_name] += 1

    print("\nOverall Time Usage Statistics:")
    for model_name in sorted(model_total_time, key=lambda x: model_total_time[x], reverse=True):
        avg_time = model_total_time[model_name] / model_call_count[model_name] if model_call_count[model_name] > 0 else 0
        print(f"{model_name}: Total time: {model_total_time[model_name]:.2f}s, Avg time per call: {avg_time:.2f}s")

    # Save overall time usage statistics
    log_folder = f"logs_with_{args.shot}shots"
    with open(f"{log_folder}/overall_time_usage.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_total_time": model_total_time,
            "model_call_count": model_call_count,
            "model_avg_time": {name: model_total_time[name] / count if count > 0 else 0 
                               for name, count in model_call_count.items()}
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()