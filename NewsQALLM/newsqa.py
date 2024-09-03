import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import TextStreamer,pipeline,AutoTokenizer


model ="ambideXtrous9/NewsQAFinetunedFlanT5s"

tokenizer = AutoTokenizer.from_pretrained(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_name1 = "sentence-transformers/all-MiniLM-L6-v2"

model_kwargs = {"device": device}

embeddings = HuggingFaceEmbeddings(model_name=model_name1, model_kwargs=model_kwargs)

vector_store = FAISS.load_local("faiss_index", 
                                embeddings,
                                allow_dangerous_deserialization=True)


streamer = TextStreamer(tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True)

pipe =  pipeline(task = 'text2text-generation',
                         model = model,
                         temperature=0.5,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                         top_p=0.25,  # select from top tokens whose probability add up to 15%
                         top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                         max_new_tokens=10,  # mex number of tokens to generate in the output
                         repetition_penalty=2.1,  # without this output begins repeating
                         do_sample = True,
                         )

llm = HuggingFacePipeline(pipeline = pipe)


prompt_template = """
                Use following piece of context to answer the question in less than 30 words.

                Context : {context}

                Question : {question}

                Answer : """


PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["context", "question"]
)


retriever = vector_store.as_retriever(search_kwargs = {"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever, 
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)


def predict(question):
    result = qa_chain(question)
    return result['result']