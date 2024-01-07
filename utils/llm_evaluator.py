import utils
from llama_index.embeddings import HuggingFaceEmbedding

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index import set_global_tokenizer
from transformers import AutoTokenizer

llm = LlamaCPP(
    model_path="/home/ivanleech/apps/github/llm/dolphin-2_6-phi-2.Q4_K_M.gguf",
    temperature=0.1,
    max_new_tokens=256,
    context_window=2048,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 0},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False,
)

set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)

# create a service context
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# load documents
documents = SimpleDirectoryReader("../assets/data").load_data()

sentence_index = utils.build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index",
)

# set up query engine
query_engine = sentence_index.as_query_engine()


### SETUP MODEL EVALUATION ###
from trulens_eval import Tru

tru = Tru()
tru.reset_database()

import numpy as np
from trulens_eval import Feedback
from trulens_eval.feedback import Groundedness
from trulens_eval import LiteLLM, TruLlama
import litellm
import pandas as pd

litellm.set_verbose = False
provider = LiteLLM(model_engine="ollama/dolphin-phi", api_base="http://localhost:11434")
context_selection = TruLlama.select_source_nodes().node.text

grounded = Groundedness(groundedness_provider=provider)

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()

f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(context_selection)
)

f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(context_selection)
    .on_output()
)

### RUN MODEL EVALUATION ###
sentence_window_engine = utils.get_sentence_window_query_engine(sentence_index)

tru_recorder = TruLlama(
    sentence_window_engine,
    app_id="App_1",
    feedbacks=[f_qa_relevance, f_qs_relevance, f_groundedness],
)
eval_questions = ["What did the author do growing up?", "Who is Paul Graham"]

for question in eval_questions:
    with tru_recorder as recording:
        sentence_window_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])


pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]

tru.get_leaderboard(app_ids=[])
tru.run_dashboard()
