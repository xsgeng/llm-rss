import json
from datetime import datetime, timezone
from pathlib import Path

import dateutil.parser
import feedparser
import ollama
import typer
from django.utils.feedgenerator import Rss201rev2Feed
from tqdm import tqdm, trange
import toml
from queue import Queue
import threading
from openai import OpenAI

from adapter import RSSAdapter

def prepare_prompt(entry):
    title = entry["title"]
    abstract = entry["abstract"]

    return f"""
    title: {title}
    abstract: {abstract}
    """

def to_bullets(text_list: list[str]):
    return "\n".join([f"- {item}" for item in text_list])


def prepare_system_prompt(config: dict):
    research_areas = to_bullets(config['research_areas'])
    excluded_areas = to_bullets(config['excluded_areas'])

    return f"""
    You are an academic paper evaluator.
    Based on the provided title, abstract, and the user's research areas, evaluate the content, including its relevance (0-9), impact (0-9), and reason (str).

    Relevance refers to the correlation with the research areas; impact assesses the value of the article, which can be high even if it's not relevant; 
    the reason should include how you assessed the relevance and impact.
    
    Reply in JSON format.
    EXAMPLE JSON OUTPUT:
    {{
        "reason": "The article ...",
        "relevance": 3,
        "impact": 5,
    }}

    User's research areas:
    {research_areas}
    Excluded areas:
    {excluded_areas}
    """


def get_ollama_reply(entry, config, model='qwen2.5:32b') -> dict:
    system_prompt = prepare_system_prompt(config)
    
    user_prompt = prepare_prompt(entry)
    
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system','content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options=dict(num_ctx=2048, temperature=0, format='json', steam=False)
    )

    
    try:
        response_parse = json.loads(response['message']['content'])
    except json.decoder.JSONDecodeError:
        response_parse = {"reason": "decode error", "relevance": 0, "impact": 0}

    return response_parse


def get_openai_reply(entry, config, model, api_key) -> dict:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/beta",
    )

    system_prompt = prepare_system_prompt(config)
    
    user_prompt = prepare_prompt(entry)

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            'type': 'json_object'
        },
        max_tokens=2048,
        temperature=0,
        stream=False
    )
    
    try:
        response_parse = json.loads(response.choices[0].message.content)
    except json.decoder.JSONDecodeError:
        response_parse = {"reason": "decode error", "relevance": 0, "impact": 0}

    return response_parse


def main(config_path: Path="config.toml"):
    config = toml.load(config_path)
    rss_path = config.get('rss_path', 'data/rss.xml')
    period = config.get('period', 24)
    relevance_threshold = config.get("relevance_threshold", 5)
    impact_threshold = config.get("impact_threshold", 3)

    model = config.get('model', None)
    if model is None:
        raise ValueError("model is not specified in the config file.")

    model_type, model_name = model.split('/')
    if model_type not in ['ollama', 'openai']:
        raise ValueError("Invalid model type. Must be 'ollama' or 'openai'.")

    api_key = config.get('api_key', None)


    new_feed = Rss201rev2Feed(
        title="Filtered RSS",
        link="myserver",
        description="Filtered arXiv RSS feed",
        language="en",
    )
    
    now = datetime.now(tz=timezone.utc)

    rss_urls = config['urls']
    recent_entries = []
    article_titles = []
    for url in rss_urls:
        rss_adapter = RSSAdapter(url)

        recent_entry = list(rss_adapter.recent_entries(hours=period))
        
        # filter duplicated
        for entry in recent_entry:
            if entry['title'] not in article_titles:
                article_titles.append(entry['title'])
            else:
                recent_entry.remove(entry)

        print(f"{len(recent_entry)} articles  to process on {url}.")

        recent_entries.extend(recent_entry)
    

    q = Queue()
    for entry in recent_entries:
        if model_type == 'ollama':
            t = threading.Thread(target=lambda: q.put(get_ollama_reply(entry, config, model=model_name)))
        elif model_type == 'openai':
            t = threading.Thread(target=lambda: q.put(get_openai_reply(entry, config, model=model_name, api_key=api_key)))
        t.start()

    for i in trange(len(recent_entries)):
        reply = q.get()
        relevance = reply['relevance']
        impact = reply['impact']

        if relevance > relevance_threshold and impact > impact_threshold:
            new_feed.add_item(
                title=entry["title"],
                link=entry["link"],
                description=f"{relevance=}\n {impact=}\n "+entry["abstract"],
                pubdate=now
            )

    if new_feed.num_items() > 0:
        with open(rss_path, "w") as f:
            new_feed.write(f, "utf-8")
    else:
        print("not updated")

if __name__ == "__main__":
    typer.run(main)

