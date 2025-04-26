import streamlit as st
import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv_path = find_dotenv()
if dotenv_path:
    logger.info(f"Found .env file at: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.error("No .env file found!")
    st.error("No .env file found in the project directory!")
    st.stop()

# Get and validate API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🔑 GROQ_API_KEY not found in .env file!")
    st.info("Please check your .env file contains:")
    st.code("GROQ_API_KEY=gsk_your_api_key_here")
    st.stop()
elif not GROQ_API_KEY.startswith("gsk_"):
    st.error("❌ Invalid API key format! Groq API keys should start with 'gsk_'")
    st.info("Please check your .env file contains a valid Groq API key")
    st.stop()

# Debug info (will only show in development)
if st.secrets.get("dev_mode"):
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"API Key found: {'Yes' if GROQ_API_KEY else 'No'}")
    st.sidebar.write(f"API Key starts with: {GROQ_API_KEY[:7]}...")

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    # Test the API key with a simple request
    test_response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": "test"}],
        model="llama3-8b-8192",
        max_tokens=5
    )
    logger.info("Successfully connected to Groq API")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    st.error(f"Failed to connect to Groq API: {str(e)}")
    st.info("Please check your API key and internet connection")
    st.stop()

@st.cache_resource
def load_rag_data():
    """Load the preprocessed RAG data"""
    try:
        # Load FAISS index
        index = faiss.read_index("vector_store.index")
        
        # Load other data
        with open("rag_data.pkl", "rb") as f:
            data = pickle.load(f)
            
        return index, data
    except FileNotFoundError:
        st.error("❌ Required data files not found!")
        st.info("Please run preprocess.py first to prepare the 911 calls data:")
        st.code("python preprocess.py")
        return None, None
    except Exception as e:
        logger.error(f"Error loading RAG data: {e}")
        st.error(f"Error loading data: {str(e)}")
        return None, None

def ask_groq_llm(system_prompt, context, user_question, model="llama3-8b-8192"):
    """Query the Groq LLM"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
        ]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.2,
            max_tokens=512
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq LLM: {e}")
        st.error(f"Error getting response from Groq: {str(e)}")
        return None

# Page config
st.set_page_config(
    page_title="911 Calls Analyzer",
    page_icon="🚨",
    layout="wide"
)

# Streamlit UI
st.title("911 Calls Analysis Bot 🚨")
st.write("Ask questions about emergency 911 calls to get data-driven insights.")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This bot analyzes 911 emergency call data to help:
    - Identify high-risk areas and times
    - Understand emergency patterns
    - Track response types and frequencies
    - Monitor temporal and spatial trends
    """)
    st.write("---")
    st.write("Example questions:")
    st.write("- What are the most common types of 911 calls?")
    st.write("- Which areas have the highest call volume?")
    st.write("- What time of day has most emergencies?")
    st.write("- How many medical emergencies in specific township?")

# Initialize the RAG system
try:
    index, data = load_rag_data()
    if data:
        embedder = data["embedder"]
        texts = data["texts"]
        metadata = data["metadata"]
        
        # Query input
        query = st.text_input("💭 Ask your question about 911 calls:", 
                            placeholder="e.g., What are the most common types of emergency calls?")
        
        # Fixed number of reference documents
        k = 3  # Using 3 reference documents for balanced context

        system_prompt = """You are a 911 calls data analyst who provides concise, direct answers from the emergency call dataset you're analyzing. Follow these critical guidelines:

1. PRIORITIZE BREVITY
   - Answer questions in the most direct way possible
   - For simple questions, limit responses to 2-3 sentences unless more detail is explicitly requested
   - Eliminate all unnecessary preambles, redundant explanations, and verbose language
   - Do not include section headings or analysis categories unless specifically asked

2. ANSWER FORMAT BASED ON QUESTION COMPLEXITY
   - For simple questions:
     * Provide a single direct statement answering the question
     * Include only the most relevant data point that supports your answer
     * Do not include methodology, confidence levels, or predictions unless specifically requested
   
   - For complex questions requiring deeper analysis:
     * Start with a 1-sentence direct answer
     * Follow with only the specific data points requested
     * Use bullet points only when presenting multiple distinct findings

3. DATA-EXCLUSIVE ANALYSIS
   - Use only patterns clearly present in the uploaded dataset
   - Do not speculate beyond what the data directly shows
   - If the data doesn't conclusively answer a question, simply state: "Based on the available data, I cannot determine [X]"

4. PREDICTIVE ANSWERS
   - Only offer predictions when explicitly asked for forecasting
   - Keep predictive statements limited to 1-2 sentences focused on the most probable outcome
   - Base predictions solely on clear patterns in the historical data

5. EXAMPLE RESPONSES:
   - Q: "When do we see most emergency calls?"
     A: "Emergency calls peak between 5-7pm on weekdays, with an average of 45 calls per hour during these times."
   
   - Q: "What's the most common emergency type?"
     A: "Medical emergencies are the most frequent, accounting for 35% of all 911 calls."

CRITICAL: The user values conciseness above all else. Never provide analytical methodology, confidence intervals, or additional context unless specifically requested.
"""

        if st.button("🔍 Analyze", type="primary"):
            if query:
                try:
                    with st.spinner("🔄 Analyzing 911 calls data..."):
                        # Generate query embedding
                        query_embedding = embedder.encode([query])[0]
                        
                        # Search in FAISS index
                        D, I = index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
                        
                        # Prepare context for LLM
                        context = "\n\n".join([texts[idx] for idx in I[0]])
                        
                        # Call Groq LLM
                        with st.spinner("🤔 Generating insights..."):
                            answer = ask_groq_llm(system_prompt, context, query)
                            
                        if answer:
                            st.markdown("### 📊 Analysis Results")
                            st.write(answer)
                        
                            # Show sources in expander
                            with st.expander("📚 View Source Data"):
                                st.write("These insights are based on the following 911 calls:")
                                for i, idx in enumerate(I[0]):
                                    st.markdown(f"**Call {i+1}** (Relevance Score: {D[0][i]:.4f})")
                                    st.info(texts[idx])
                                    st.write(f"Metadata: {metadata[idx]}")
                                    st.markdown("---")
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    st.error(f"Error processing your query: {str(e)}")
            else:
                st.warning("Please enter a question to analyze.")
except Exception as e:
    logger.error(f"Error initializing app: {e}")
    st.error(f"Error initializing the application: {str(e)}")
    st.info("Please make sure you've run preprocess.py first and all required files are present.")
