import feedparser
from datetime import datetime, timezone, timedelta
import dateutil.parser
from typing import Iterator

class BaseRSSAdapter:
    def __init__(self, rss_url):
        self.rss_url = rss_url
        self.feed = feedparser.parse(rss_url)

    @property
    def entries(self) -> Iterator[dict]:
        for entry in self.feed.entries:
            yield {
                "title": self._get_entry_title(entry),
                "link": self._get_entry_link(entry),
                "abstract": self._get_entry_abstract(entry),
                "updated": self._get_entry_updated(entry),
                "authors": self._get_entry_authors(entry)
            }

    def __iter__(self):
        """
        This method is used to make the class iterable.
        """
        return self.entries()

    def recent_entries(self, hours=24) -> Iterator[dict]:
        def is_recent(entry):
            return datetime.now(timezone.utc) - entry['updated'] < timedelta(hours=hours)

        return filter(is_recent, self.entries)

    def _get_entry_title(self, entry):
        return entry['title']

    def _get_entry_link(self, entry):
        return entry['link']

    def _get_entry_abstract(self, entry):
        return entry['summary']

    def _get_entry_updated(self, entry):
        return dateutil.parser.parse(entry['updated'])

    def _get_entry_authors(self, entry):
        if 'authors' in entry and entry['authors']: 
            return ','.join([author['name'] for author in entry['authors']])
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
        # Nature abstract is after <p></p> block in entry['summary']
        return entry['summary'].split('</p>')[1]

    def _get_entry_updated(self, entry):
        # Nature has no zoneinfo, use utc
        entry_time = dateutil.parser.parse(entry['updated'])
        entry_time = entry_time.replace(tzinfo=timezone.utc)
        return entry_time

class ArxivAdapter(BaseRSSAdapter):
    def _get_entry_abstract(self, entry):
        return entry['summary'].split('Abstract: ')[1]
    
    
class BioRxivAdapter(BaseRSSAdapter):
    def _get_entry_updated(self, entry):
        entry_time = dateutil.parser.parse(entry['updated'])
        entry_time = entry_time.replace(tzinfo=timezone.utc)
        return entry_time

class APSAdapter(BaseRSSAdapter):
    def _get_entry_abstract(self, entry):
        if '<p>' in entry['summary']:
            return entry['summary'].split('<p>')[1].split('</p>')[0]
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

