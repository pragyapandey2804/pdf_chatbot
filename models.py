import os
from langchain_ollama import OllamaEmbeddings, ChatOllama

class Models:
    def __init__(self):
        self.embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        self.model_ollama = ChatOllama(
            model="llama3",
            temperature=0,
        )