# LLM-RSS

## Introduction

LLM-RSS is a tool that reads title and abstract from RSS sources like Nature, Arxiv, etc., filters them based on specific research areas or keywords provided by users, and then generates a new RSS feed in xml. Modify the prompt as needed. The generated xml file can then be hosted via nginx and accessed via Zotero, for example. Unlike keyword matching, LLM-RSS uses large language models (LLMs) to understand the content, providing a more contextually relevant feed.

## Installation

To install and run LLM-RSS, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xsgeng/llm-rss.git
   cd llm-rss
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the project:**
   - Copy the `config.d/comfig.toml.example` to `config.d/config.toml`.
   - Modify the `config.toml` file to include your RSS feed URLs, research areas, and other settings.
   - The program sequentially processes each toml file.


## Supported RSS Providers

LLM-RSS currently supports the following RSS providers:

- Nature
- Arxiv
- APS (American Physical Society)
- BioRxiv
- Cell
- AIP (American Institute of Physics)
- IOP (Institute of Physics)

## Use with Cron

LLM-RSS is intended to be used along with a cron job to periodically update the filtered RSS feed. Here is an example of how to set up a cron job:

1. **Open the crontab editor:**
   ```bash
   crontab -e
   ```

2. **Add a new cron job:**
   ```bash
   0 0 * * * /usr/bin/python3 /path/to/llm-rss/main.py
   ```
   This cron job will run the script every day at 0:00.


## Hosting via Nginx
A `docker-compose.yaml` is provided to host the generated RSS feed via Nginx:
```
docker-compose up -d
```
The RSS feed will be available at http://localhost:8080/rss/rss.xml depending on your setup.

## Feed to Zotero

Add http://localhost:8080/rss/rss.xml to your Zotero RSS feed. Then you can read and access your papers in Zotero.