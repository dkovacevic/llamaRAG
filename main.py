from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from vector import retriever

model = OllamaLLM(model="llama3.2")

# 2) Build a RetrievalQA chain that uses “stuff” (i.e., “concat all documents and feed them together”)
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",          # “stuff” means “dump all retrieved docs into one prompt”
    retriever=retriever,
    return_source_documents=True  # if you want to inspect which pages were used
)

template = """
You are a senior software engineer. I want you to help junior colleagues to better understand our documentation
Here are some relevant confluence pages.
Here is the question to answer: What is LexoRank?
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("question: ")
    print("\n\n")
    if question == "q":
        break

    # This single call will:
    #   • retrieve the 5 best docs,
    #   • automatically concatenate their .page_content,
    #   • insert that text into a default prompt template,
    #   • and call the LLM.
    output = qa_chain({"query": question})

    # output["result"] is the LLM’s answer
    # output["source_documents"] is a list of the actual Document objects used
    print("Answer:\n", output["result"])
    print("\n\nSources:")
    for doc in output["source_documents"]:
        print(" •", doc.metadata.get("title", doc.id))