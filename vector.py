from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from confluence import fetch_confluence_pages
from confluence import extract_text_from_storage

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    print("Fetching pages from Confluence...")
    pages = fetch_confluence_pages()

    documents = []
    ids = []
    seen_ids = set()

    for page in pages:
        page_id = page["id"]

        # Skip if we've already added this page_id
        if page_id in seen_ids:
            continue
        seen_ids.add(page_id)

        title = page.get("title", "")
        storage_value = page.get("body", {}).get("storage", {}).get("value", "")
        content_text = extract_text_from_storage(storage_value)

        metadata = {
            "title": title,
            "space": page.get("space", {}).get("key", ""),
            "version": page.get("version", {}).get("number", None),
            "last_updated": page.get("version", {}).get("when", "")
        }

        doc = Document(
            page_content=title + "\n" + content_text,
            metadata=metadata,
            id=page_id
        )
        ids.append(page_id)
        documents.append(doc)

vector_store = Chroma(
    collection_name="confluence",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)