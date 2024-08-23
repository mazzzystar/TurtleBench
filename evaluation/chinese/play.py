import os

import autogen
from dotenv import load_dotenv

from evaluate import stories
from prompt import player_system_prompt, simple_system_prompt

load_dotenv()

config_list = [{'model': 'gpt-4o-mini', 'api_key': os.getenv("OPENAI_API_KEY")}]

llm_config = {"config_list": config_list}

user_proxy = autogen.UserProxyAgent(
    name="Proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("完全正确") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },
)

task = """请通过最少得猜测提问，还原出<汤底>，提问数量限制在8次以内"""


def reflection_message(recipient, messages, sender, config):
    return f"{recipient.chat_messages_for_summary(sender)[-1]['content']}"


def run(story):
    prompt = simple_system_prompt.replace("{surface}", story["surface"]).replace("{bottom}", story["bottom"])
    referee = autogen.AssistantAgent(
        name="Referee",
        llm_config={"config_list": config_list},
        system_message=prompt,
    )
    prompt = player_system_prompt.replace("{surface}", story["surface"]).replace("{bottom}", story["bottom"])
    player = autogen.AssistantAgent(
        name="Player",
        llm_config={"config_list": config_list},
        system_message=prompt,
    )
    user_proxy.register_nested_chats(
        [{"recipient": referee, "message": reflection_message, "summary_method": "last_msg", "max_turns": 1}],
        trigger=player,
    )
    res = user_proxy.initiate_chat(recipient=player, message=task, max_turns=8, summary_method="last_msg")
    count = 0
    for chat in res.chat_history:
        print(f"'{chat['name']}: {chat['content']}'")
        count += 1
        if chat['content'] == '完全正确':
            return True, int(count/2)
    return False, int(count/2)


if __name__ == '__main__':
    _story = stories[1]
    print(run(_story))
