import os
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from together import Together
import requests  # Add this import at the top of the file

load_dotenv()

MAX_TOKENS = 5

class ModelType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    MINIMAX = "minimax"

import os
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from together import Together
import requests

load_dotenv()

MAX_TOKENS = 5

class Model(Enum):
    GPT_35_TURBO = ("GPT-3.5-Turbo", ModelType.OPENAI, "gpt-3.5-turbo")
    KIMI_CHAT = ("Kimi-Chat", ModelType.OPENAI, "moonshot-v1-8k")
    GPT_4O_MINI = ("GPT-4o-mini", ModelType.OPENAI, "gpt-4o-mini")
    GPT_4O_20240806 = ("GPT-4o-2024-08-06", ModelType.OPENAI, "gpt-4o-2024-08-06")
    GPT_O1_MINI_20240912 = ("o1-mini-2024-09-12", ModelType.OPENAI, "o1-mini-2024-09-12")
    GPT_O1_PREVIEW_20240912 = ("o1-preview-2024-09-12", ModelType.OPENAI, "o1-preview-2024-09-12")
    CLAUDE_3_HAIKU = ("Claude-3-Haiku", ModelType.ANTHROPIC, "claude-3-haiku-20240307")
    CLAUDE_35_SONNET = ("Claude-3.5-Sonnet", ModelType.ANTHROPIC, "claude-3-5-sonnet-20240620")
    LLAMA_31_405B = ("Llama-3.1-405b", ModelType.TOGETHER, "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
    LLAMA_31_70B = ("Llama3.1-70b", ModelType.TOGETHER, "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
    QWEN2_72B = ("Qwen2-72B-Instruct", ModelType.TOGETHER, "Qwen/Qwen2-72B-Instruct")
    DOUBAO_4K = ("Doubao-4k", ModelType.OPENAI, "ep-20240802142948-6vvc7")
    DEEPSEEK = ("Deepseek-Chat", ModelType.OPENAI, "deepseek-chat")

    def __init__(self, display_name, model_type, model_id):
        self.display_name = display_name
        self.type = model_type
        self.model_id = model_id

def is_o1_model(model: Model):
    return model in [Model.GPT_O1_MINI_20240912, Model.GPT_O1_PREVIEW_20240912]

def get_model_config(model: Model):
    base_config = {
        "temperature": 0.0,
        "top_p": 1,
        "max_tokens": MAX_TOKENS,
    }
    
    config = {
        ModelType.OPENAI: lambda: {
            "apiKey": (os.getenv("OPENAI_API_KEY") if model not in [Model.DEEPSEEK, Model.DOUBAO_4K, Model.KIMI_CHAT] else
                       os.getenv("DEEPSEEK_API_KEY") if model == Model.DEEPSEEK else
                       os.getenv("DOUBAO_API_KEY") if model == Model.DOUBAO_4K else
                       os.getenv("MOONSHOT_API_KEY")),
            "baseURL": ("https://api.openai.com/v1" if model not in [Model.DEEPSEEK, Model.DOUBAO_4K, Model.KIMI_CHAT] else
                        "https://api.deepseek.com/v1" if model == Model.DEEPSEEK else
                        "https://ark.cn-beijing.volces.com/api/v3" if model == Model.DOUBAO_4K else
                        "https://api.moonshot.cn/v1"),
            "model": model.model_id,
            "timeout": 40 if is_o1_model(model) else 10,
            **(base_config if not is_o1_model(model) else {})
        },
        ModelType.ANTHROPIC: lambda: {
            **base_config,
            "apiKey": os.getenv("ANTHROPIC_API_KEY"),
            "model": model.model_id,
        },
        ModelType.TOGETHER: lambda: {
            **base_config,
            "apiKey": os.getenv("TOGETHER_API_KEY"),
            "model": model.model_id,
            "repetition_penalty": 1,
            "stop": [""]
        },
        ModelType.MINIMAX: lambda: {
            **base_config,
            "apiKey": os.getenv("MINIMAX_API_KEY"),
            "groupId": os.getenv("MINIMAX_GROUP_ID"),
            "model": model.model_id,
        }
    }
    
    return config.get(model.type, lambda: ValueError(f"Unsupported model type: {model.type}"))()

def call_api(model: Model, prompt: str, user_input: str):
    config = get_model_config(model)
    
    try:
        if model.type == ModelType.OPENAI:
            client = OpenAI(api_key=config["apiKey"], base_url=config["baseURL"])
            
            if is_o1_model(model):
                # For o1 models, combine prompt and user_input
                combined_input = f"{prompt}\n\nUser: {user_input}"
                messages = [{"role": "user", "content": combined_input}]
            else:
                # For other OpenAI models, keep the original format
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ]
            
            api_params = {
                "model": config["model"],
                "messages": messages,
            }
            
            if not is_o1_model(model):
                api_params.update({
                    "max_tokens": config["max_tokens"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                })
            
            response = client.chat.completions.create(**api_params)
            return response.choices[0].message.content
        
        elif model.type == ModelType.ANTHROPIC:
            client = Anthropic(api_key=config["apiKey"])
            message = client.messages.create(
                model=config["model"],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                system=prompt,
                messages=[{"role": "user", "content": user_input}]
            )
            return message.content[0].text
        
        elif model.type == ModelType.TOGETHER:
            client = Together(api_key=config["apiKey"])
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                repetition_penalty=config["repetition_penalty"],
                stop=config["stop"],
                stream=False
            )
            return response.choices[0].message.content
        
        elif model.type == ModelType.MINIMAX:
            url = f"https://api.minimax.chat/v1/text/chatcompletion_v2?GroupId={config['groupId']}"
            headers = {
                "Authorization": f"Bearer {config['apiKey']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": config["model"],
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"],
                "top_p": config["top_p"]
            }
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        else:
            raise ValueError(f"Unsupported model type: {model.type}")
    except Exception as e:
        print(f"Error in call_api for model {model.name}: {str(e)}")
        return None