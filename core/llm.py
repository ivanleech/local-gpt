from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp, CTransformers


def get_embedding_llm():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_chat_llm():
    # Note: You have to download this model from Huggingface and place locally
    return LlamaCpp(
        model_path="/home/ivanleech/apps/github/llm/phi-2.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False,
    )

    # Updating from CTransformers to LlamaCpp as former does not support latest LLM models
    # return CTransformers(
    #     model="../../llm/zephyr-7b-beta.Q4_K_M.gguf",
    #     # model="../../llm/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf",
    #     model_type="mistral",
    #     lib="avx2",
    # )
