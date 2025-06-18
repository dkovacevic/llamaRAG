from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

#from vector import retriever
from repoRAG import retriever

model = OllamaLLM(model="gemma3")

# 2) Build a RetrievalQA chain that uses “stuff” (i.e., “concat all documents and feed them together”)
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",          # “stuff” means “dump all retrieved docs into one prompt”
    retriever=retriever,
    return_source_documents=True  # if you want to inspect which pages were used
)

template = """
You are a helpful technical assistant for a software company. You answer questions using only the provided Confluence documentation snippets. 

Below are several excerpts from the company's Confluence wiki pages. These may include documentation, architectural diagrams, how-to guides, onboarding notes, release notes, or process manuals. Use these excerpts as your only source of truth.

When answering, always:
- Base your answer solely on the retrieved Confluence content. If you can't find a direct answer, say so clearly.
- Provide concise, clear, and professional responses suitable for engineers or business users.
- If multiple relevant snippets are found, synthesize the answer from all, citing the snippet(s) when appropriate.
- Never make up information that is not present in the provided snippets.

----
Question:
{question}

----
Relevant Confluence wiki snippets:
{context}

----
Answer:

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
    output = qa_chain.invoke({"query": question})

    # output["result"] is the LLM’s answer
    # output["source_documents"] is a list of the actual Document objects used
    print("Answer:\n", output["result"])
    print("\n\nSources:")
    for doc in output["source_documents"]:
        print(" •", doc.metadata.get("title", doc.id))