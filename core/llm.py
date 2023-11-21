from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import CTransformers


def get_embedding_llm():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_chat_llm():
    # Note: You have to download this model from Huggingface and place locally
    return CTransformers(
        model="../llm/zephyr-7b-beta.Q4_K_M.gguf",
        model_type="mistral",
        lib="avx2",
    )
