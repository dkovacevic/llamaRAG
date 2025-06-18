import os
import subprocess
import json
import sys
import time
import re

from pathlib import Path
from langchain import hub
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.ollama import Ollama
from langchain.chains.question_answering import load_qa_chain

REPO_URL = "https://gitlab.witt-gruppe.eu/ewp/scm/sova/sova-rest.git"
LOCAL_PATH = "repo"
CHROMA_DIR = "chroma_db"
MAX_RETRIES = 5

def clone_or_update_repo(repo_url: str, local_path: str):
    if not os.path.exists(local_path):
        subprocess.run(["git", "clone", repo_url, local_path], check=True)
    else:
        subprocess.run(["git", "-C", local_path, "pull"], check=True)

def find_java_files(base_path: str) -> List[Path]:
    base = Path(base_path)
    return [
        f for f in base.rglob("*.java")
    ]

def load_documents(files: List[Path]) -> List[Document]:
    docs = []
    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"path": str(filepath)}))
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
    return docs

def parse_json_from_llm_output(output: str) -> Any:
    """Extract JSON from LLM output robustly."""
    # Remove Markdown code blocks
    codeblock_match = re.search(r"```(?:json)?(.*?)```", output, re.DOTALL)
    if codeblock_match:
        output = codeblock_match.group(1)
    # Remove leading/trailing junk
    output = output.strip()
    # Remove leading/trailing junk before/after curly braces
    json_match = re.search(r"({.*})", output, re.DOTALL)
    if json_match:
        output = json_match.group(1)
    return json.loads(output)

def main():
    # Step 1: Clone repo
    clone_or_update_repo(REPO_URL, LOCAL_PATH)

    # Step 2: Find relevant Java files
    java_files = find_java_files(LOCAL_PATH)
    if not java_files:
        print("No Java files found in '-rest' directories.")
        sys.exit(0)

    documents = load_documents(java_files)
    if not documents:
        print("No Java documents could be loaded.")
        sys.exit(1)

    # Step 3: Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunked_docs = []
    for doc in documents:
        chunked_docs.extend(splitter.split_documents([doc]))

    texts = [d.page_content for d in chunked_docs]
    metadatas = [d.metadata for d in chunked_docs]

    # Step 4: Embeddings & Vector DB
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma.from_texts(
        texts,
        embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR
    )
    vector_store.persist()

    # Step 5: RetrievalQA
    llm = Ollama(model="gemma3")
    combine_docs_chain = load_qa_chain(llm)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA(combine_documents_chain=combine_docs_chain, retriever=retriever)

    # Step 6: Prompt
    analysis_prompt = (
        "You are an assistant analyzing Spring Boot code snippets. "
        "You need to identify every database table used in the code. "
        "The best way is to look for annotations like @Table or @Entity. "
        "Also include the columns of these tables. These usually are properties marked with @Column or @Id. "
        "Output strictly valid JSON with this schema:\n"
        "{\n"
        "  \"Entities\": {\n"
        "    \"<table-name>\": {\n"
        "      \"columns\": [\n"
        "        {\"column name\": \"<column-name>\", \"data type\": \"<data-type>\"}\n"
        "      ]\n"
        "    }\n"
        "  },\n"
        "  \"FilesUsed\": [\"<file1.java>\", ...]\n"
        "}\n"
        "Do not wrap the JSON in a code block."
    )

    # Step 7: Analysis with Retry
    print("Running analysis...")
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = qa_chain.run(analysis_prompt)
            # Extract JSON safely
            report = parse_json_from_llm_output(result)
            # Save report and files used
            with open("inter_service_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print("Report written to inter_service_report.json")
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            last_error = e
            time.sleep(attempt * 2)  # Exponential backoff
    else:
        print("All retries failed. Last error:", last_error)
        print("Last model output was:")
        print(result)

# -----------------------------------------------------------------------------
# 8. Basic Unit Tests for Splitter
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

