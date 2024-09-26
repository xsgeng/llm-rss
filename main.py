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

def prepare_prompt(entry):
    title = entry.title
    abstract = entry.summary#.split('\n')[1].split(':')[1]

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
        options=dict(temperature=0, format='json')
    )

    
    try:
        response_parse = json.loads(response['message']['content'])
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


    new_feed = Rss201rev2Feed(
        title="Filtered RSS",
        link="myserver",
        description="Filtered arXiv RSS feed",
        language="en",
    )
    
    now = datetime.now(tz=timezone.utc)
    def filter_recent(entry):
        entry_time = dateutil.parser.parse(entry['updated'])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        return (now - entry_time).total_seconds() < period * 3600

    rss_urls = config['urls']
    recent_entries = []
    for url in rss_urls:
        online_feed = feedparser.parse(url)

        recent_entry = list(filter(filter_recent, online_feed.entries))

        print(f"{len(recent_entry)} articles  to process on {url}.")

        recent_entries.extend(recent_entry)
    

    q = Queue()
    for entry in recent_entries:
        t = threading.Thread(target=lambda: q.put(get_ollama_reply(entry, config, model=model)))
        t.start()

    for i in trange(len(recent_entries)):
        reply = q.get()
        relevance = reply['relevance']
        impact = reply['impact']

        if relevance > relevance_threshold and impact > impact_threshold:
            new_feed.add_item(
                title=entry.title,
                link=entry.link,
                description=f"{relevance=}\n {impact=}\n "+entry.summary,
                pubdate=now
            )

    if new_feed.num_items() > 0:
        with open(rss_path, "w") as f:
            new_feed.write(f, "utf-8")
    else:
        print("not updated")

if __name__ == "__main__":
    typer.run(main)

