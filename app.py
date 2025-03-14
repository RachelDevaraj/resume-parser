import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import PyPDF2
import os

# Get API Key
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
if not api_key:
    st.error("API Key is missing. Set GOOGLE_GEMINI_API_KEY as an environment variable.")
    st.stop()

# Initialize Gemini API
try:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
except Exception as e:
    st.error(f"Error initializing Gemini API: {e}")
    st.stop()

# Function to extract key details from a resume
def parse_resume(resume_text):
    """Extracts key details from a resume."""
    prompt = f"""
    Extract key details such as:
    - Name
    - Contact information
    - Skills
    - Experience
    - Education
    from the following resume:
    {resume_text}
    """
    try:
        response = model.invoke(prompt)
        
        # Extract only the text content, ignoring metadata
        if isinstance(response, dict) and "text" in response:
            return response["text"]  # Show only the extracted resume details
        else:
            return response  # Fallback in case response format differs
    except Exception as e:
        return f"Error processing resume: {e}"

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
    except Exception as e:
        return f"Error extracting text: {e}"

# Streamlit UI
st.title("Resume Parser")
uploaded_file = st.file_uploader("Upload your resume", type=["txt", "pdf"])

if uploaded_file:
    try:
        resume_text = (
            extract_text_from_pdf(uploaded_file)
            if uploaded_file.name.endswith(".pdf")
            else uploaded_file.read().decode("utf-8", errors="ignore").strip()
        )

        if resume_text:
            st.write(parse_resume(resume_text))
        else:
            st.error("Failed to extract text. The file might be empty or unreadable.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
