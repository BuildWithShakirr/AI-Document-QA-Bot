import streamlit as st
from groq import Groq
import PyPDF2
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API Key not found! Please check your .env file.")
    st.stop()

# Initialize the Groq Client
client = Groq(api_key=api_key)

# UI Layout
st.title("📄 AI Document Q&A Bot")
st.write("Upload a PDF document and ask questions based on its content using Llama 3 (via Groq).")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    with st.spinner("Reading and processing PDF..."):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Speed & Token Limit: Limit text to first 5000 characters for fast processing
        text = text[:5000]
    
    st.success("PDF successfully processed! You can now ask questions.")
    
    # User input for question
    question = st.text_input("Ask a question about the document:")
    
    if st.button("Generate Answer"):
        if question:
            prompt = f"Here is the document text:\n\n{text}\n\nBased on this text, answer this question: {question}"
            
            st.write("### 🤖 AI Response:")
            
            try:
                # Groq API Call with Streaming
                stream = client.chat.completions.create(
                    model="openai/gpt-oss-20b", # Ye Groq ka sabse fast model hai
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based strictly on the provided document text."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True 
                )
                
                # Function to yield text chunks for Streamlit
                def generate_chunks():
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                
                # Display the streamed response
                st.write_stream(generate_chunks)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question first!")