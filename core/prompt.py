import os
import pinecone

from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from .llm import get_chat_llm, get_embedding_llm
from dotenv import load_dotenv

load_dotenv()


pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
embedding_llm = get_embedding_llm()
chat_llm = get_chat_llm()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    docsearch = Pinecone.from_existing_index(
        embedding=embedding_llm,
        index_name=INDEX_NAME,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat_llm, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm("What is your name?"))