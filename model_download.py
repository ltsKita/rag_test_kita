from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

model_name = "elyza/Llama-3-ELYZA-JP-8B"

# モデル格納先ディレクトリを指定
model_dir = "model"

# トークナイザーとモデルを指定ディレクトリにキャッシュ
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir)

# Embeddingsのセットアップ
embeddings_model_name = "intfloat/multilingual-e5-large"

# Embeddingsのモデルを事前にキャッシュ
embedding_model = SentenceTransformer(embeddings_model_name, cache_folder=model_dir)

# Embeddingsのセットアップ
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

def get_tokenizer():
    return tokenizer

def get_model():
    return model

def get_embeddings():
    return embeddings

