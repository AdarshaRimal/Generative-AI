# static prompts in langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

st.header('Football Player Info')
user_input = st.text_input('Enter your prompt')

if st.button('summarize'):
    result = model.invoke(user_input)  #static prompt
    st.write(result.content)
