import os
import requests
from bs4 import BeautifulSoup

# Confluence configuration: set these environment variables or replace with your credentials
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "https://confluence.atlassian.com")
USERNAME = os.getenv("CONFLUENCE_USERNAME", "")
API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")


def fetch_confluence_pages():
    """
    Fetch all Confluence pages in the instance by paginating through the REST API.
    Returns a list of page JSON objects.
    """
    pages = []
    start = 0
    limit = 50
    auth = (USERNAME, API_TOKEN)

    while True:
        params = {
            "cql": "type=page",
            "start": start,
            "limit": limit,
            "expand": "body.storage,version,space"
        }
        url = f"{CONFLUENCE_BASE_URL}/rest/api/content/search"
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        print("Got results from Confluence")

        pages.extend(results)
        if len(results) < limit:
            break
        start += limit

    return pages


def extract_text_from_storage(storage_body: str) -> str:
    """
    Given a Confluence storage-format HTML string, extract plain text.
    """
    soup = BeautifulSoup(storage_body, "html.parser")
    return soup.get_text(separator="\n")

