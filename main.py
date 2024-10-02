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
    Based on the provided title, abstract, and the user's research areas, evaluate the content, including its relevance (0-9), impact (0-9).

    Relevance refers to the correlation with the research areas; impact assesses the value of the article, which can be high even if it's not relevant.
     
    Please reply how you assessed the relevance of the article to the reasearch areas and the potential impact of the article.
    Your reply should be less than 500 words. Reply can be short if it is totally irrelevant.

    User's research areas:
    {research_areas}
    Excluded areas:
    {excluded_areas}
    """
    
    
def prepare_json_prompt():
    return """
    Reply in JSON format.
    EXAMPLE JSON OUTPUT:
    {{
        "relevance": 3,
        "impact": 5,
    }}
"""


def get_ollama_reply(entry, config, model, base_url='localhost:11434') -> dict:
    client = ollama.Client(host=base_url)
    system_prompt = prepare_system_prompt(config)
    user_prompt = prepare_prompt(entry)
    json_prompt = prepare_json_prompt()
    
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    response = client.chat(
        model=model,
        messages=messages,
        stream=False,
        options=dict(num_ctx=2048, temperature=0)
    )['message']['content']
    # print(response)
    
    messages += [{"role": "assistant", "content": response},
                {"role": "user", "content": json_prompt}]
    json_reply = client.chat(
        model=model,
        messages=messages,
        format='json',
        stream=False,
        options=dict(num_ctx=2048, temperature=0)
    )['message']['content']
    
    try:
        response_parse = json.loads(json_reply)
    except json.decoder.JSONDecodeError:
        response_parse = {"reason": "decode error", "relevance": 0, "impact": 0}

    return response_parse


def get_openai_reply(entry, config, base_url, model, api_key) -> dict:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    system_prompt = prepare_system_prompt(config)
    
    user_prompt = prepare_prompt(entry)
    json_prompt = prepare_json_prompt()

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=0,
        stream=False
    ).choices[0].message.content
    
    messages += [{"role": "assistant", "content": response},
                {"role": "user", "content": json_prompt}]
                 
    json_reply = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            'type': 'json_object'
        },
        max_tokens=2048,
        temperature=0,
        stream=False
    ).choices[0].message.content
    
    try:
        response_parse = json.loads(json_reply)
    except json.decoder.JSONDecodeError:
        response_parse = {"reason": "decode error", "relevance": 0, "impact": 0}

    return response_parse


def main(config_path: Path="config.toml", dryrun: bool=False):
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

    base_url = config.get('base_url', None)
    if model_type == 'ollama' and base_url is None:
        base_url = 'localhost:11434'
    if model_type == 'openai' and base_url is None:
        raise ValueError("base_url is not specified in the config file.")
    
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
            t = threading.Thread(target=lambda: q.put(get_ollama_reply(entry, config, model=model_name, base_url=base_url)))
        elif model_type == 'openai':
            t = threading.Thread(target=lambda: q.put(get_openai_reply(entry, config, model=model_name, base_url=base_url, api_key=api_key)))
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

    if new_feed.num_items() > 0 and not dryrun:
        with open(rss_path, "w") as f:
            new_feed.write(f, "utf-8")
    else:
        print("not updated")
        
def _main(config_dir: Path='config.d', config_path: Path=None):
    if config_path is None:
        for config_path in config_dir.glob("*.toml"):
            print(f"{config_path}:")
            main(config_path=config_path)
    else:
        main(config_path=config_path)

if __name__ == "__main__":
    typer.run(_main)

