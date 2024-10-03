import json


with open("results_20240925141021.json") as f:
    results = json.load(f)["logs"]

with open("sanitized.json") as f:
    sanitized = json.load(f)["logs"]

complement_sanitized = []
for item in results:
    if not any([item["guess"] == x["guess"] for x in sanitized]):
        complement_sanitized.append(item)


def get_token_usage(results: list) -> tuple[int, int]:
    prompt_tokens = sum([x["prompt_tokens"] for x in results])
    completion_tokens = sum([x["completion_tokens"] for x in results])
    return prompt_tokens / len(results), completion_tokens / len(results)


def print_completion_tokens(results: list):
    for x in results:
        print(x["completion_tokens"], end=", ")
    print('\n')

print(f"""{len(results)=}
{len(sanitized)=}
{len(complement_sanitized)=}
""")

print(
    f"""{get_token_usage(results)=}
{get_token_usage(sanitized)=}
{get_token_usage(complement_sanitized)=}
"""
)

for x in [results, sanitized, complement_sanitized]:
    print_completion_tokens(x)
