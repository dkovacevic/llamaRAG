import os
import requests
from bs4 import BeautifulSoup

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "https://confluence.atlassian.com")
USERNAME = os.getenv("CONFLUENCE_USERNAME", "")
API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
SPACE_KEY = "ERPCRM"
PARENT_ID = "169651392"
pages_dir = "pages"
os.makedirs(pages_dir, exist_ok=True)

def extract_text_from_storage(storage_value):
    soup = BeautifulSoup(storage_value, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def make_safe_filename(title, page_id):
    import re
    safe = re.sub(r"[^\w\- ]", "_", title).strip().replace(" ", "_")
    return f"{safe[:48]}_{page_id}"

def fetch_confluence_pages():
    start = 0
    limit = 50
    cql = f'space="{SPACE_KEY}" AND ancestor={PARENT_ID}'
    base_url = f"{CONFLUENCE_BASE_URL}/rest/api/content/search"
    auth = (USERNAME, API_TOKEN)
    total_fetched = 0
    all_pages = []

    url = base_url
    while True:
        params = {
            "cql": cql,
            "start": start,
            "limit": limit,
            "expand": "body.storage"
        }
        response = requests.get(url, params=params, auth=auth)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        print(f"Got {len(results)} with start={start}, limit={limit}")

        # If no more results, exit the loop
        if not results:
            break

        # Process and write files for current batch
        new_in_batch = 0
        for page in results:
            title = page.get("title", "")
            page_id = page.get("id", "")
            storage_value = page.get("body", {}).get("storage", {}).get("value", "")
            content_text = extract_text_from_storage(storage_value)
            if not title or not content_text.strip():
                continue
            filename = f"{make_safe_filename(title, page_id)}.txt"
            filepath = os.path.join(pages_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content_text)
            print(f"Wrote {filepath}")
            all_pages.append(page)
            total_fetched += 1
            new_in_batch += 1

        # Use Confluence's paging: follow next link if present
        next_link = data.get("_links", {}).get("next")
        if not next_link:
            break
        url = CONFLUENCE_BASE_URL + next_link
        params = {}  # Only send params on the first request

    print(f"Fetched and wrote {len(all_pages)} pages.")
    return all_pages

