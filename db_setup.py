import PyPDF2
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from model_download import get_embeddings

embeddings = get_embeddings()

# PDFファイルのパス
pdf_path = "data/関西電力_美浜発電所公開情報.pdf"

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# PDFからテキストを抽出
text = extract_text_from_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_text(text)

chroma_directory = "db"
vectorstore = Chroma.from_texts(
    texts=splits,
    embedding=embeddings,
    persist_directory=chroma_directory
)
