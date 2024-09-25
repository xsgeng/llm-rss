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

rss_urls = [
    "http://feeds.aps.org/rss/recent/prl.xml",
    "http://feeds.aps.org/rss/recent/pra.xml",
    "http://feeds.aps.org/rss/recent/prx.xml",
    "http://rss.arxiv.org/rss/physics.plasm-ph",
    "http://rss.arxiv.org/rss/physics.comp-ph",
]

def prepare_prompt(entry):
    title = entry.title
    abstract = entry.summary#.split('\n')[1].split(':')[1]

    return f"""
    title: {title}
    abstract: {abstract}
    """

def prepare_system_prompt(config: dict):
    research_areas = "\n".join([f"- {area}" for area in config['research_areas']])
    excluded_areas = "\n".join([f"- {area}" for area in config['excluded_areas']])

    return f"""
    你是一个学术论文评分者。
    依据提供的题目和摘要和用户的研究领域，对内容进行评价，包括相关性(relevance,0-9)、影响力(impact, 0-9)、理由(reason, str)，以json格式回复。
    相关性是与与研究领域的相关性；影响力评估文章的价值，即便完全不相关仍可以具有高影响力；理由包括你如何评估相关性与影响力。
    EXAMPLE JSON OUTPUT:
    {{
        "reason": "The article ...",
        "relevance": 3,
        "impact": 5,
    }}

    用户的研究领域：
    {research_areas}
    排除的领域：
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
        options=dict(temperature=0)
    )

    
    try:
        response_parse = json.loads(response['message']['content'])
    except json.decoder.JSONDecodeError:
        response_parse = {"reason": "decode error", "relevance": 0, "impact": 0}

    return response_parse


def main(config_path: Path="config.toml"):
    config = toml.load(config_path)
    rss_path = config['rss_path']
    period = config['period']

    new_feed = Rss201rev2Feed(
        title="Filtered RSS",
        link="myserver",
        description="Filtered arXiv RSS feed",
        language="en",
    )

    def filter_recent(entry):
        entry_time = dateutil.parser.parse(entry['updated'])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        return (datetime.now(timezone.utc) - entry_time).total_seconds() < period * 3600

    rss_urls = config['urls']
    for url in rss_urls:
        online_feed = feedparser.parse(url)

        recent_entry = list(filter(filter_recent, online_feed.entries))

        print(f"{len(recent_entry)} articles on {url} to process.")

        q = Queue()
        for entry in recent_entry:
            t = threading.Thread(target=lambda: q.put(get_ollama_reply(entry, config)))
            t.start()

        for i in trange(len(recent_entry)):
            reply = q.get()
            relevance = reply['relevance']
            impact = reply['impact']

            if relevance > 5:
                new_feed.add_item(
                    title=entry.title,
                    link=entry.link,
                    description=f"{relevance=}\n {impact=}\n "+entry.summary,
                    pubdate=datetime.now()
                )

    if new_feed.num_items() > 0:
        with open(rss_path, "w") as f:
            new_feed.write(f, "utf-8")
    else:
        print("not updated")

if __name__ == "__main__":
    typer.run(main)

