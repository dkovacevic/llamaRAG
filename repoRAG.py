import os
import subprocess
import json
import sys
import bs4

from pathlib import Path
from langchain import hub
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict

# LangChain components
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.ollama import Ollama
from langchain.chains.question_answering import load_qa_chain


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
REPO_URL = "https://github.com/malckal/vizsgaevkonyv.git"
LOCAL_PATH = "repo"
CHROMA_DIR = "chroma_db"

# -----------------------------------------------------------------------------
# Dependency Check
# -----------------------------------------------------------------------------
# If hub.load or other imports fail, instruct user to install
try:
    # Test hub functionality
    _ = hub
except Exception as e:
    print("Error: LangChain hub not available. Please install latest LangChain packages:")
    print("  pip install langchain langchain-core langchain-text-splitters langchain-community chromadb ollama")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1. Clone or Update Repo
# -----------------------------------------------------------------------------
if not os.path.exists(LOCAL_PATH):
    subprocess.run(["git", "clone", REPO_URL, LOCAL_PATH], check=True)
else:
    subprocess.run(["git", "-C", LOCAL_PATH, "pull"], check=True)

# -----------------------------------------------------------------------------
# 2. Load Java files as Documents
# -----------------------------------------------------------------------------
java_files = list(Path(LOCAL_PATH).rglob("*.java"))
documents: List[Document] = []
for filepath in java_files:
    try:
        text = filepath.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"path": str(filepath)}))
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")

if not documents:
    print("No Java files found to process.")
    sys.exit(0)

# -----------------------------------------------------------------------------
# 3. Split Documents into Chunks
# -----------------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunked_docs: List[Document] = []
for doc in documents:
    splits = splitter.split_documents([doc])
    chunked_docs.extend(splits)

texts = [d.page_content for d in chunked_docs]
metadatas = [d.metadata for d in chunked_docs]

# -----------------------------------------------------------------------------
# 4. Embeddings and Chroma Ingestion via hub
# -----------------------------------------------------------------------------
config = {
    "type":"ollama",
    "component":"embeddings",
    "model":"mxbai-embed-large"
}
config_str = json.dumps(config)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma.from_texts(
    texts,
    embeddings,
    metadatas=metadatas,
    persist_directory=CHROMA_DIR
)
vector_store.persist()

# -----------------------------------------------------------------------------
# 5. Setup RetrievalQA with Ollama LLM
# -----------------------------------------------------------------------------
config = {
    "type":"ollama",
    "component":"model",
    "model":"llama3"
}
config_str = json.dumps(config)


llm = Ollama(model="llama3")
combine_docs_chain = load_qa_chain(llm)
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA(
    combine_documents_chain=combine_docs_chain,  # use default chain
    retriever=retriever,
)

# -----------------------------------------------------------------------------
# 6. Analysis Prompt Definition
# -----------------------------------------------------------------------------
analysis_prompt = (
    "You are an assistant analyzing Spring Boot code snippets. "
    "You need to identify every database table used in the code."
    "Also include the columns of these tables. These are usually marked with @Column or @Id"
    "Only take the directory ending with '-rest' in consideration."
    "The best way to look for annotations like @Table or @Entity."
    "Output strictly valid JSON with this schema:\n"
    "{\n"
    "  \"Entities\": {\n"
    "    \"<table-name>\": {\n"
    "      \"columns\": [\n"
    "        {\n"
    "          \"column name\": \"<column-name>\",\n"
    "          \"data type\": \"<data-type>\",\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  }\n"
    "}\n"
    "\n"
    "Also print all the file names you use for your analysis"
)

# -----------------------------------------------------------------------------
# 7. Run Analysis and Save Report
# -----------------------------------------------------------------------------
print("Running analysis...")
for i in range(3):
    result = qa_chain.run(analysis_prompt)

    try:
        report = json.loads(result)
        with open("inter_service_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Report written to inter_service_report.json")
    except json.JSONDecodeError:
        print("Failed to parse JSON from model output.\nOutput was:\n", result)

# -----------------------------------------------------------------------------
# 8. Basic Unit Tests for Splitter
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import unittest

    class TestSplitter(unittest.TestCase):
        def test_single_chunk(self):
            text = "a " * 100
            docs = splitter.split_documents([Document(page_content=text)])
            self.assertEqual(len(docs), 1)

        def test_multi_chunk(self):
            text = "word " * 600
            docs = splitter.split_documents([Document(page_content=text)])
            self.assertGreaterEqual(len(docs), 2)

    unittest.main()
