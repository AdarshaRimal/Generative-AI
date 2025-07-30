import streamlit as st
import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load HuggingFace token from .env or secrets
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# Initialize LLM + Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
model = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    huggingfacehub_api_token=HF_TOKEN,
)
chat_llm = ChatHuggingFace(llm=model)

# Extract video ID from URL
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

# Fetch transcript using proper API usage
def fetch_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en"])
        raw = fetched.to_raw_data()
        text = " ".join(chunk["text"] for chunk in raw)
        return text
    except TranscriptsDisabled:
        return "No captions available for this video."
    except Exception as e:
        return f"Unexpected error: {e}"

# Build the LangChain-based QA system
def process_video(video_id):
    text = fetch_transcript(video_id)
    if text.startswith("Unexpected error") or text == "No captions available for this video.":
        st.error(text)
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = PromptTemplate.from_template("""
    You are a helpful assistant. Use the context to answer user questions.

    Context:
    {context}

    Question:
    {question}
    """)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=False,
    )

    return qa_chain

# Streamlit UI
st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬")
st.title("🎬 Chat with a YouTube Video")

youtube_url = st.text_input("Paste a YouTube Video URL:")

if youtube_url:
    video_id = extract_video_id(youtube_url)
    if video_id:
        st.success(f"Video ID: {video_id}")
        qa_chain = process_video(video_id)
        if qa_chain:
            st.session_state.qa_chain = qa_chain
    else:
        st.error("Invalid YouTube URL format.")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question about the video:")
    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({"question": user_question})
            st.markdown(f"**💬 Answer:** {response['answer']}")
