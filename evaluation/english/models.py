import os
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from together import Together

load_dotenv()

MAX_TOKENS = 4

class ModelType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"

class Model(Enum):
    GPT_35_TURBO = ("GPT-3.5-Turbo", ModelType.OPENAI, "gpt-3.5-turbo")
    GPT_4O_MINI = ("GPT-4o-mini", ModelType.OPENAI, "gpt-4o-mini")
    GPT_4O_20240806 = ("GPT-4o-2024-08-06", ModelType.OPENAI, "gpt-4o-2024-08-06")
    GPT_O1_MINI_20240912 = ("o1-mini-2024-09-12", ModelType.OPENAI, "o1-mini-2024-09-12")
    GPT_O1_PREVIEW_20240912 = ("o1-preview-2024-09-12", ModelType.OPENAI, "o1-preview-2024-09-12")
    CLAUDE_3_HAIKU = ("Claude-3-Haiku", ModelType.ANTHROPIC, "claude-3-haiku-20240307")
    CLAUDE_35_SONNET = ("Claude-3.5-Sonnet", ModelType.ANTHROPIC, "claude-3-5-sonnet-20240620")
    LLAMA_3_70B = ("Llama3.1-70b", ModelType.TOGETHER, "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
    LLAMA_31_405B = ("Llama-3.1-405b", ModelType.TOGETHER, "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
    Qwen2_72B = ("Qwen2-72B-Instruct", ModelType.TOGETHER, "Qwen/Qwen2-72B-Instruct")
    DEEPSEEK_V25 = ("Deepseek-V2.5", ModelType.OPENAI, "deepseek-chat")
    # Add other models as needed

    def __init__(self, display_name, model_type, model_id):
        self.display_name = display_name
        self.type = model_type
        self.model_id = model_id

def is_o1_model(model: Model):
    return model in [Model.GPT_O1_MINI_20240912, Model.GPT_O1_PREVIEW_20240912]

def get_model_config(model: Model):
    base_config = {
        "temperature": 0.0,
        "top_p": 0.7,
        "max_tokens": MAX_TOKENS,
    }
    
    if model.type == ModelType.OPENAI:
        config = {
            "apiKey": os.getenv("OPENAI_API_KEY") if model != Model.DEEPSEEK_V25 else os.getenv("DEEPSEEK_API_KEY"),
            "baseURL": "https://api.openai.com/v1" if model != Model.DEEPSEEK_V25 else "https://api.deepseek.com/v1",
            "model": model.model_id,
            "timeout": 40 if is_o1_model(model) else 10,
        }
        if not is_o1_model(model):
            config.update(base_config)
        return config
    elif model.type == ModelType.ANTHROPIC:
        return {
            **base_config,
            "apiKey": os.getenv("ANTHROPIC_API_KEY"),
            "model": model.model_id,
        }
    elif model.type == ModelType.TOGETHER:
        return {
            **base_config,
            "apiKey": os.getenv("TOGETHER_API_KEY"),
            "model": model.model_id,
            "stop": ["<|eot_id|>"]
        }
    else:
        raise ValueError(f"Unsupported model type: {model.type}")

def call_api(model: Model, prompt: str, user_input: str):
    config = get_model_config(model)
    
    try:
        if model.type == ModelType.OPENAI:
            client = OpenAI(api_key=config["apiKey"], base_url=config["baseURL"])
            
            # Combine prompt and user_input for all OpenAI models
            combined_input = f"{prompt}\n\nUser: {user_input}"
            messages = [{"role": "user", "content": combined_input}]
            
            api_params = {
                "model": config["model"],
                "messages": messages,
            }
            
            # Only add these parameters if they're not o1 models
            if not is_o1_model(model):
                api_params.update({
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_tokens": config["max_tokens"],
                })
            
            response = client.chat.completions.create(**api_params, timeout=config["timeout"])
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("The response does not contain choices.")
        
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
                stop=config["stop"],
                stream=False
            )
            return response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported model type: {model.type}")
    except Exception as e:
        print(f"Error in call_api for model {model.name}: {str(e)}")
        return None