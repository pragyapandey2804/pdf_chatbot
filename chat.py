from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain  
from langchain_chroma import Chroma
from models import Models
from langchain.chains.retrieval import create_retrieval_chain


#Initialize the model
models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama

#initialize the vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db", ##locally data will be saved here
)

#Define the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant. Answer the question based only on the information in the document."),
        ("human", "Use the user question {input} to answer the question. Use onlt the {context} to answer the questions.")
    ]
)

#define retrieval chain
retriever = vector_store.as_retriever(kwargs={"k":10})
combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)
retriever_chain = create_retrieval_chain(
    retriever, combine_docs_chain
)

#Main loop
def main():
    while True:
        query = input("User (or type 'q', 'quit', or 'exit' to end the chat): ")
        if query.lower() in ["q", "quit", "exit"]:
            break

        result = retriever_chain.invoke({"input": query})
        print("Assistant:", result["answer"], "\n\n")

#run the loop
if __name__ == "__main__":
    main()        