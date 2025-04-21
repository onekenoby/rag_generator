import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env
load_dotenv()

# Fetch the Gemini API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Set the environment variable for the Google GenAI client
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)

# Prompt template with persona, tone, and depth
template = """
You are answering as a {persona}. Your tone should be {tone}, and the explanation should have {depth} depth.

Question: {question}

Generate a prompt to ask an LLM to provide the best possible answer under these conditions.
"""

prompt = ChatPromptTemplate.from_template(template)

# LangChain chain
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.set_page_config(page_title="Smart Prompt Generator", layout="centered")
st.title("Smart Prompt Generator")
st.write("Create dynamic prompts for LLMs based on persona, tone, and depth.")

# Inputs
question = st.text_area("Enter your question:", "How does machine learning work?")
persona = st.selectbox("Persona:", ["Student", "Developer", "Executive", "Teacher", "Data Analyst"])
tone = st.selectbox("Tone:", ["Simple", "Technical", "Persuasive", "Neutral", "Encouraging"])
depth = st.selectbox("Depth:", ["Basic", "Intermediate", "Advanced"])

if st.button("Generate Prompt"):
    with st.spinner("Generating..."):
        prompt_text = chain.run({
            "persona": persona,
            "tone": tone.lower(),
            "depth": depth.lower(),
            "question": question
        })

        st.subheader("Generated Prompt for LLM")
        st.code(prompt_text, language="text")
