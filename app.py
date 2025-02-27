import streamlit as st
import os
import shutil
from chat import retriever_chain
from ingest import ingest_file  # Importing the ingestion function

# Constants
HEIGHT = 500
TITLE = "My Chatbot"
ICON = ":robot:"
DATA_FOLDER = "./data"

# Function to handle file uploads and store them in the data folder
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to process the uploaded PDF
def process_uploaded_file(uploaded_file):
    file_path = save_uploaded_file(uploaded_file)
    ingest_file(file_path)  # Ingest the file into ChromaDB
    st.success(f"✅ {uploaded_file.name} has been added and processed!")

# Initialize the conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Streamlit UI setup
st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide")
st.header(TITLE)

# File uploader section
st.subheader("📂 Upload PDF Files")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    process_uploaded_file(uploaded_file)

# Chatbot interaction
st.subheader("💬 Chat with your PDFs")
message_container = st.container(border=True, height=HEIGHT)

def generate_message(user_input):
    response = retriever_chain.invoke({"input": user_input})
    answer = response["answer"]

    st.session_state.conversation.append({
        "user": user_input,
        "assistant": answer
    })

    for entry in st.session_state.conversation:
        message_container.chat_message("user").write(entry["user"])
        message_container.chat_message("assistant").write(entry["assistant"])

if prompt := st.chat_input("Enter your question..."):
    generate_message(prompt)
