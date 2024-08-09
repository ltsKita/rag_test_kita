from model_download import get_tokenizer, get_model
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

tokenizer = get_tokenizer()
model = get_model()

def get_llm():
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0
        )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# print(llm("今日の天気は？"))