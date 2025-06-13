from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from confluence import fetch_confluence_pages
from confluence import extract_text_from_storage

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Configure the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,      # You can tune this, try 500-2000 for LLMs with ~4k context
    chunk_overlap=200     # A little overlap helps with context at boundaries
)

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

        # ---- Use splitter here ----
        # Combine title and content for context
        to_split = title + "\n" + content_text
        split_docs = splitter.create_documents([to_split])

        for idx, chunk in enumerate(split_docs):
            # Give each chunk a unique ID
            chunk_id = f"{page_id}-{idx}"
            doc = Document(
                page_content=chunk.page_content,
                metadata=metadata,
                id=chunk_id
            )
            ids.append(chunk_id)
            documents.append(doc)

vector_store = Chroma(
    collection_name="confluence",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)