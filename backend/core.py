import os
from typing import Any
import dotenv
import pprint

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

dotenv.load_dotenv(verbose=True)

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
)

INDEX_NAME = os.getenv("INDEX_NAME")


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(run_llm(query="What is LangChain?"))
