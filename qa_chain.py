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

"""

# ベクトルストアの利用例
query = "特定重大事故等対処施設について教えて。"
results = loaded_vectorstore.similarity_search(query,k=5)

# 結果を表示
for i, result in enumerate(results):
    print(f"Result {i+1}: {result}")

"""

# 検索用の関数
retriever = loaded_vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = PromptTemplate.from_template("""
以下のコンテキストを使用して、質問に答えてください。
コンテキスト: {context}

質問: {question}

回答：""")


# LCEL チェーンの構築
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "発電用原子炉の炉心の、通常運転時の熱的制限値の設定はどのように定められていますか？"

result = qa_chain.invoke(query)
print("回答:", result)