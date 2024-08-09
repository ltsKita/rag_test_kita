from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from model_download import get_embeddings
from llm_pipeline import get_llm

# エンベディングモデルの取得
embeddings = get_embeddings()

# llmパイプラインの取得
llm = get_llm()

# 永続化ディレクトリの指定
chroma_directory = "db"

# 永続化されたベクトルストアを読み込む
loaded_vectorstore = Chroma(
    persist_directory=chroma_directory,
    embedding_function=embeddings, 
)

# 検索用の関数
retriever = loaded_vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template("""
以下のコンテキストを使用して、質問に答えてください。
コンテキスト: {context}

質問: {question}

回答：""")

def format_docs(docs):
    return "\n\n".join(f"Result {i+1}: {doc.page_content}" for i, doc in enumerate(docs))

def extract_answer(full_response: str) -> str:
    if "回答：" in full_response:
        return full_response.split("回答：", 1)[1].strip()
    return full_response.strip()

# LCEL チェーンの構築
def qa_chain(question: str):
    context = retriever.invoke(question)
    formatted_context = format_docs(context)

    qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | extract_answer
    )

    answer = qa_chain.invoke(question)
    
    return {
        "question": question,
        "answer": answer,
        "sources": formatted_context
    }

# Streamlit用の関数
def get_qa_response(user_input: str):
    response = qa_chain(user_input)
    return response