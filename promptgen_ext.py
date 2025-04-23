import os
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize Gemini LLM and memory
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.4)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# Prompt template
template = """
You are answering as a {persona}. Your tone should be {tone}, and the explanation should have {depth} depth.

Question: {question}

Generate a prompt to ask an LLM to provide the best possible answer under these conditions.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Feedback file setup
feedback_file = 'feedback_data.csv'
if not os.path.exists(feedback_file):
    pd.DataFrame(columns=["timestamp", "prompt", "persona", "tone", "depth", "feedback"]).to_csv(feedback_file, index=False)

# Streamlit UI setup
st.set_page_config(page_title="Adaptive PromptCraft AI", layout="centered")
st.title("✨ Adaptive PromptCraft AI")
st.write("Generate dynamic, personalized prompts enhanced by real-time feedback and context.")

# Initialize session state for storing generated prompt details
if 'prompt_generated' not in st.session_state:
    st.session_state['prompt_generated'] = False

# User Input section
question = st.text_area("📝 Enter your question:", "How does machine learning work?")
col1, col2 = st.columns(2)

with col1:
    persona = st.selectbox("🎭 Persona:", ["Student", "Developer", "Executive", "Teacher", "Data Analyst"])
    custom_persona = st.text_input("Or enter custom persona:")

with col2:
    tone = st.selectbox("🎨 Tone:", ["Simple", "Technical", "Persuasive", "Neutral", "Encouraging"])
    custom_tone = st.text_input("Or enter custom tone:")

depth = st.selectbox("📚 Depth:", ["Basic", "Intermediate", "Advanced"])

# Generate prompt button
if st.button("🚀 Generate Prompt"):
    with st.spinner("Generating personalized prompt..."):
        final_persona = custom_persona if custom_persona else persona
        final_tone = custom_tone.lower() if custom_tone else tone.lower()

        prompt_text = chain.run({
            "persona": final_persona,
            "tone": final_tone,
            "depth": depth.lower(),
            "question": question
        })

        # Save generated prompt in session state
        st.session_state['prompt_text'] = prompt_text
        st.session_state['final_persona'] = final_persona
        st.session_state['final_tone'] = final_tone
        st.session_state['depth'] = depth.lower()
        st.session_state['prompt_generated'] = True

# Display prompt if generated
if st.session_state.get('prompt_generated', False):
    st.subheader("🔖 Generated Prompt")
    st.code(st.session_state['prompt_text'], language="text")

    # Feedback mechanism (outside generation button)
    feedback = st.radio("👍 Rate Prompt Quality:", ["👍 Great", "👌 Okay", "👎 Needs Improvement"], horizontal=True)

    if st.button("Submit Feedback"):
        feedback_df = pd.read_csv(feedback_file)
        new_entry = {
            "timestamp": datetime.now(),
            "prompt": st.session_state['prompt_text'],
            "persona": st.session_state['final_persona'],
            "tone": st.session_state['final_tone'],
            "depth": st.session_state['depth'],
            "feedback": feedback
        }
        feedback_df = pd.concat([feedback_df, pd.DataFrame([new_entry])], ignore_index=True)
        feedback_df.to_csv(feedback_file, index=False)
        st.success("Feedback saved. Thank you! 🙌")
        st.session_state['prompt_generated'] = False  # Reset after feedback submission

# Analytics dashboard
st.divider()
st.subheader("📊 Analytics Dashboard")

if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file)

    if not feedback_df.empty:
        st.write("### Feedback Summary")
        feedback_counts = feedback_df['feedback'].value_counts().reset_index()
        feedback_counts.columns = ['Feedback', 'Count']
        st.bar_chart(feedback_counts.set_index('Feedback'))

        st.write("### Persona Popularity")
        persona_counts = feedback_df['persona'].value_counts()
        st.bar_chart(persona_counts)

        st.write("### Tone Usage")
        tone_counts = feedback_df['tone'].value_counts()
        st.bar_chart(tone_counts)

        st.write("### Depth Preferences")
        depth_counts = feedback_df['depth'].value_counts()
        st.bar_chart(depth_counts)
    else:
        st.info("No feedback data yet. Start using the app and provide feedback to see analytics here!")
else:
    st.error("Feedback file not found. Generate some prompts and provide feedback first.")
