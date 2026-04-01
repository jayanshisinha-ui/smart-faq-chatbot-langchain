import tkinter as tk
from tkinter import scrolledtext
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Loading chatbot... Please wait.")

# Load FAQ data
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

print("Chatbot Ready!")

# ================= MEMORY =================

chat_history = []

def save_history(user_msg, bot_msg):
    chat_history.append(("You", user_msg))
    chat_history.append(("Bot", bot_msg))

    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"You: {user_msg}\n")
        f.write(f"Bot: {bot_msg}\n\n")

# ================= GUI =================

window = tk.Tk()
window.title("Smart FAQ Chatbot")
window.geometry("700x600")
window.configure(bg="#343541")

# Chat area
chat_area = scrolledtext.ScrolledText(
    window,
    wrap=tk.WORD,
    width=80,
    height=25,
    font=("Arial", 11),
    bg="#444654",
    fg="white"
)

chat_area.pack(padx=10, pady=10)

# Function to send message
def send_message(event=None):

    question = user_input.get()

    if question.strip() == "":
        return

    # Show user message
    chat_area.insert(
        tk.END,
        f"\nYou: {question}\n",
        "user"
    )

    docs = vector_store.similarity_search(
        question,
        k=1
    )

    answer_text = docs[0].page_content

    lines = answer_text.split("\n")

    if len(lines) > 1:
        answer = lines[1]
    else:
        answer = answer_text

    # Show bot message
    chat_area.insert(
        tk.END,
        f"Bot: {answer}\n",
        "bot"
    )

    save_history(question, answer)

    chat_area.see(tk.END)

    user_input.delete(0, tk.END)

# Input frame
input_frame = tk.Frame(window, bg="#343541")
input_frame.pack(pady=5)

# Entry box
user_input = tk.Entry(
    input_frame,
    width=50,
    font=("Arial", 12),
    bg="#40414F",
    fg="white",
    insertbackground="white"
)

user_input.pack(side=tk.LEFT, padx=5)

user_input.bind("<Return>", send_message)

# Send button
send_button = tk.Button(
    input_frame,
    text="Send",
    command=send_message,
    bg="#19C37D",
    fg="white",
    width=10
)

send_button.pack(side=tk.LEFT)

# Clear chat
def clear_chat():
    chat_area.delete("1.0", tk.END)

clear_button = tk.Button(
    input_frame,
    text="Clear",
    command=clear_chat,
    bg="#EF4444",
    fg="white",
    width=10
)

clear_button.pack(side=tk.LEFT, padx=5)

# Welcome message
chat_area.insert(
    tk.END,
    "Smart FAQ Chatbot Initialized!\nAsk your question about python , Dbms etc.\n\n"
)

window.mainloop()