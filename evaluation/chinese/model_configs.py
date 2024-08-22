import os

from dotenv import load_dotenv

MAX_TOKENS = 5
# Load environment variables
load_dotenv()

# Define the models and their configurations
models = [
    {
        "name": "DEEPSEEK",
        "config": {
            "apiKey": os.getenv("DEEPSEEK_API_KEY"),
            "baseURL": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1
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
            "top_p": 1
        },
        "type": "openai"
    },
    {
        "name": "Kimi-Chat",
        "config": {
            "apiKey": os.getenv("MOONSHOT_API_KEY"),
            "baseURL": "https://api.moonshot.cn/v1",
            "model": "moonshot-v1-8k",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1
        },
        "type": "openai"
    },
    {
        "name": "GPT-4o",
        "config": {
            "apiKey": os.getenv("OPENAI_API_KEY"),
            "baseURL": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1
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
            "top_p": 1
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
            "top_p": 1,
            "repetition_penalty": 1,
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
            "top_p": 1,
            "repetition_penalty": 1,
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
            "top_p": 1,
            "repetition_penalty": 1,
            "stop": ["<|im_start|>", "<|im_end|>"]
        },
        "type": "together"
    },
    {
        "name": "Doubao-4k",
        "config": {
            "apiKey": os.getenv("DOUBAO_API_KEY"),
            "baseURL": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "ep-20240802142948-6vvc7",  # Replace with the actual endpoint ID if different
            "maxTokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1
        },
        "type": "openai"
    },
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
    {
        "name": "MiniMax-ABAB6.5s",
        "config": {
            "groupId": os.getenv("MINIMAX_GROUP_ID"),
            "apiKey": os.getenv("MINIMAX_API_KEY"),
            "model": "abab6.5s-chat",
            "maxTokens": MAX_TOKENS,
            "temperature": 0.01,  # must be (0, 1]
            "top_p": 1
        },
        "type": "minimax"
    },
]
