import configparser

from openai import OpenAI
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

MAX_TOKENS = 5

# Load configurations from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Build a dictionary of model configurations
model_configs = {}
for model_name in config.sections():
    model_configs[model_name] = {
        'model': config.get(model_name, 'model'),
        'base_url': config.get(model_name, 'base_url'),
        'api_key': config.get(model_name, 'api_key'),
        'temperature': 0.0,
        'top_p': 0.9,
        'max_tokens': MAX_TOKENS,
    }


def get_model_config(model_name):
    if model_name in model_configs:
        return model_configs[model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in configuration.")


def is_o1_model(model_name):
    return 'o1' in model_name.lower()


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def call_api(model_name, prompt, user_input):
    config = get_model_config(model_name)

    try:
        if is_o1_model(model_name):
            # For o1 models, combine prompt and user_input
            combined_input = f"{prompt}\n\nUser: {user_input}"
            messages = [{"role": "user", "content": combined_input}]
            params = {
                "model": config["model"],
                "messages": messages,
            }
            # o1 models may not support additional parameters
        else:
            # For other OpenAI models, use standard parameters
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
            params = {
                "model": config["model"],
                "messages": messages,
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"],
                "top_p": config["top_p"],
            }

        # Initialize OpenAI client
        client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"]
        )

        # Call the API
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        # prompt_tokens = response.usage.prompt_tokens
        # input_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        return content, total_tokens

    except Exception as e:
        print(f"Error in call_api for model {model_name}: {str(e)}")
        return None


if __name__ == '__main__':
    # Example usage
    prompt = "You are a helpful assistant."
    user_input = "how are you?"
    model_name = "Llama_3_1_70B"

    response = call_api(model_name, prompt, user_input)
    print(response)
