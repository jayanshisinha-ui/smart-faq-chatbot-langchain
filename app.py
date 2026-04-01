import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

print("Loading chatbot... Please wait.")

# Load FAQ file
with open("faq_data_full.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text
text_splitter = CharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20
)

texts = text_splitter.split_text(text)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vector_store = FAISS.from_texts(
    texts,
    embeddings
)

print("\n✅ Chatbot Ready!")
print("Type 'exit' to stop.\n")

# Chat loop
while True:

    question = input("Ask your question: ")

    if question.lower() == "exit":
        break

    docs = vector_store.similarity_search(
        question,
        k=1
    )

    print("\nAnswer:")
    print(docs[0].page_content)
    print()