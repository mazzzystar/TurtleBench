import json
import sys
from datetime import datetime

from openai import OpenAI


def request(query: str, model: str = sys.argv[1]) -> tuple[str, int, int]:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": query}]
    )
    response_str = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    return response_str, prompt_tokens, completion_tokens


TEMPLATE = """你是一个游戏的裁判，这个游戏会给玩家展示<汤面>，并告诉你<汤底>，你需要根据<汤面>和<汤底>理解整个故事。玩家会根据<汤面>进行猜测，你需要判断玩家的猜测是否正确，请回答指定三种答案：对、错、不知道；同时给出解释。

## 判定规则
   - 玩家提出的猜测正确，或者答案是肯定的：请只回答"对"。
   - 玩家提出的猜测错误，或者答案是否定的：请只回答"错"。
   - 玩家提出的猜测，从<汤面>和<汤底>找不到答案，并且也无法通过推理得出此结论：请只回答"不知道"。

## 注意
1. 玩家只能看到<汤面>，所以他是基于<汤面>进行猜测的，例如：玩家问“他喝的不是海龟汤”，是在问<汤面>中他喝的是不是海龟汤，即使<汤底>中他曾经喝过其他的汤，你也应该判定他在<汤面>中喝的是否是海龟汤。
2. 凡是无法从提供的故事中得出的结论，都应该回答"不知道"，例如：玩家提出的猜测是关于故事中的细节，而这些细节并没有在故事中提到，也无法通过推理得出此结论，那么你应该回答"不知道"。

## 题目内容
### 汤面
{surface}

### 汤底
{bottom}

现在，请判断以下玩家猜测:"""

with open("stories.json") as f:
    stories = json.load(f)

with open("specific_error_cases_zh_shot0_7.json") as f:
    guesses = json.load(f)


results = {
    "meta": {
        "model_name": sys.argv[1],
        "prompt_template": TEMPLATE,
    },
    "logs": [],
}
for idx, item in enumerate(guesses):
    story_title = item["story_title"]
    the_story = [x for x in stories if x["title"] == story_title][0]
    surface, bottom = the_story["surface"], the_story["bottom"]
    guess = item["input"]
    query = TEMPLATE.format(surface=surface, bottom=bottom) + guess
    model_response, prompt_tokens, completion_tokens = request(query)
    print(f"[{idx}] ", model_response)
    results["logs"].append(
        {
            "story": the_story,
            "guess": guess,
            "ground_truth": item["ground_truth"],
            "model_response": model_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    )

with open(f"results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
