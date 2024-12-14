import google.generativeai as genai
import streamlit as st
import pdfplumber 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure the API key
API_KEY = "AIzaSyCY0WrBTdfZgNJqrL2jDB1azlMeBCVHpIk"
genai.configure(api_key=API_KEY)

model_name = "gemini-1.5-flash"
model = genai.GenerativeModel(model_name)

# Function to fetch responses using Gemini
def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching response: {e}"

# Function to chunk text with overlap
def chunk_text_with_overlap(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to extract text from PDF left-to-right, row by row
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            rows = page.extract_text().split('\n')
            for row in rows:
                text += row.strip() + " "  # Maintain left-to-right order
    return text.strip()

# Function to find the most relevant chunk
def find_relevant_chunk(query, chunks):
    vectorizer = TfidfVectorizer().fit_transform(chunks + [query])
    similarity = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    
    # Debug: Print similarity scores
    logging.info(f"Similarity Scores: {similarity}")
    most_relevant_index = similarity.argmax()
    return chunks[most_relevant_index], similarity[most_relevant_index]

# Streamlit setup
st.set_page_config(page_title="CV Retriever Q/A ChatBot", layout="wide")
st.header("CV Retriever ChatBot")

# File upload
uploaded_file = st.file_uploader("Upload your CV (PDF format)", type="pdf")

if uploaded_file:
    with st.spinner("Processing CV..."):
        cv_text = extract_text_from_pdf(uploaded_file) 
        chunks = chunk_text_with_overlap(cv_text, chunk_size=200, overlap=30)
    st.success("CV processed successfully!")

    # Initialize session state for question history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar for history
    with st.sidebar:
        st.header("Question History")
        if st.session_state.history:
            for i, (q, a) in enumerate(st.session_state.history):
                st.write(f"**Q{i+1}:** {q}")
                st.write(f"**A{i+1}:** {a}")
                st.write("---")
        else:
            st.write("No questions asked yet.")

    # Input and Submit
    st.subheader("Ask Questions About the CV")
    user_query = st.text_input("Enter your question", placeholder="e.g., What is the candidate's education?")
    submit_query = st.button("Submit")

    if submit_query and user_query.strip():
        with st.spinner("Fetching response..."):
            # Find the most relevant chunk
            relevant_chunk, similarity_score = find_relevant_chunk(user_query, chunks)

            # Debug: Display the selected chunk
            logging.info(f"Relevant Chunk: {relevant_chunk}, Similarity Score: {similarity_score}")

            # Generate a Gemini response using the relevant chunk
            prompt = f"Based on this CV information: {relevant_chunk}\nAnswer the question: {user_query}"
            response = get_gemini_response(prompt)

            # Add to history
            st.session_state.history.append((user_query, response))

            # Display the response
            st.subheader("Response")
            st.write(response)
else:
    st.write("Please upload a CV to begin.")


