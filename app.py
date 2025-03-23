import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_desc_vector], resume_vectors).flatten()

    # Convert similarity scores to percentage
    return [round(score * 100, 2) for score in cosine_similarities]

# Streamlit App
st.title("📄 Resume Screener - AI-Powered ATS")

# Job description input
st.header("💼 Job Description")
job_description = st.text_area("Enter the job description")

# Resume file uploader
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Process resumes
if uploaded_files and job_description:
    st.header("📊 Resume Ranking Results")

    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display results in a DataFrame
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Matching Percentage (%)": scores})
    results = results.sort_values(by="Matching Percentage (%)", ascending=False)

    # Show ranked resumes
    st.dataframe(results)