from datetime import datetime, timedelta, timezone
from typing import Iterator

import dateutil.parser
import feedparser
import requests
from bs4 import BeautifulSoup

from pydantic import BaseModel, field_validator, HttpUrl, AwareDatetime

class ArticleInfo(BaseModel):
    title: str
    link: HttpUrl
    abstract: str
    updated: datetime
    authors: str

    @field_validator("updated", mode='before')
    @classmethod
    def ensure_tzinfo(cls, value: str):
        if isinstance(value, str):
            # 尝试解析字符串为datetime对象
            value = dateutil.parser.parse(value)
        
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

class BaseRSSAdapter:
    def __init__(self, rss_url):
        self.rss_url = rss_url
        self.feed = feedparser.parse(rss_url)

    @property
    def articles(self) -> Iterator[ArticleInfo]:
        for entry in self.feed.entries:
            yield ArticleInfo(
                title=self._get_entry_title(entry),
                link=self._get_entry_link(entry),
                abstract=self._get_entry_abstract(entry),
                updated=self._get_entry_updated(entry),
                authors=self._get_entry_authors(entry)
            )

    def __iter__(self):
        """
        This method is used to make the class iterable.
        """
        return self.articles

    def recent_articles(self, hours=24) -> Iterator[ArticleInfo]:
        def is_recent(articleinfo):
            return datetime.now(timezone.utc) - articleinfo.updated < timedelta(hours=hours)

        return filter(is_recent, self.articles)

    def _get_entry_title(self, entry) -> str:
        return entry['title']

    def _get_entry_link(self, entry) -> str:
        return entry['link']

    def _get_entry_abstract(self, entry) -> str:
        return entry['summary']

    def _get_entry_updated(self, entry) -> str:
        return entry['updated']

    def _get_entry_authors(self, entry) -> str:
        if 'authors' in entry and entry['authors']: 
            return ', '.join([author['name'] for author in entry['authors']])
        return ''

class RSSAdapter(BaseRSSAdapter):
    def __new__(cls, rss_url):
        if 'nature.com' in rss_url:
            return NatureAdapter(rss_url)
        elif 'arxiv.org' in rss_url:
            return ArxivAdapter(rss_url)
        elif 'aps.org' in rss_url:
            return APSAdapter(rss_url)
        elif 'aip.org' in rss_url:
            return AIPAdapter(rss_url)
        elif 'iop.org' in rss_url:
            return IOPAdapter(rss_url)
        elif 'biorxiv.org' in rss_url:
            return BioRxivAdapter(rss_url)
        elif 'cell.com' in rss_url:
            return CellAdapter(rss_url)
        else:
            raise ValueError("Unsupported RSS feed URL")

class NatureAdapter(BaseRSSAdapter):
    def _get_entry_abstract(self, entry):
        # Nature abstract is after <p></p> block in entry.summary
        return entry.summary.split('</p>')[1]


class ArxivAdapter(BaseRSSAdapter):
    def _get_entry_abstract(self, entry):
        return entry.summary.split('Abstract: ')[1]
    
    
class BioRxivAdapter(BaseRSSAdapter):
    pass

class APSAdapter(BaseRSSAdapter):
    def _get_entry_abstract(self, entry):
        if '<p>' in entry.summary:
            return entry.summary.split('<p>')[1].split('</p>')[0]
        else:
            return ''

class AIPAdapter(BaseRSSAdapter):
    pass

class IOPAdapter(BaseRSSAdapter):
    pass

class CellAdapter(BaseRSSAdapter):
    def _get_entry_updated(self, entry):
        entry_time = dateutil.parser.parse(entry['updated'])
        entry_time = entry_time.replace(tzinfo=timezone.utc)
        return entry_time

