import streamlit as st
import subprocess
import pandas as pd
import numpy as np
import os
import faiss
import base64
from openai import OpenAI


# Initialize OpenAI client
client = OpenAI(
    api_key="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",  # Replace with your API key
    base_url="https://integrate.api.nvidia.com/v1"
)

# File paths
metadata_file = "metadata.npy"
index_file = "faiss_index"

# Formatting our CSV data
def generate_text_representation(row):
    author_name = row["Author's Name"]
    title = row["Title"]
    publisher = row["Publisher / Place"]
    year = row["Year"]
    return f"{author_name} - {title} - {publisher} - {year}"

# Creating Embeddings and storing it in faiss DB
def create_faiss_database(csv_file):

    df = pd.read_csv(csv_file)
    df["text_representation"] = df.apply(generate_text_representation, axis=1)

    embeddings = []
    for text in df["text_representation"].tolist():
        response = client.embeddings.create(
            input=[text],
            model="nvidia/nv-embedqa-e5-v5",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        embeddings.append(response.data[0].embedding)

    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))

    np.save(metadata_file, df["text_representation"].tolist())
    faiss.write_index(index, index_file)
    return index, df["text_representation"].tolist()


# Similarity Search 
def search_faiss(query, top_k=3):
    query_response = client.embeddings.create(
        input=[query],
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    query_embedding = np.array([query_response.data[0].embedding])
    distances, indices = index.search(query_embedding, top_k)
    metadata = np.load(metadata_file, allow_pickle=True)
    results = [(metadata[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

    print(results)
    return results

# Book Description automatic creation
def summarize_book(book_info):
        prompt = (
            f"Provide a concise and clear description for the following book: {book_info}. "
            "Avoid introductory phrases and focus on the core content."
        )
        response = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.3,
            max_tokens=200,
            stream=False,
        )
        description = response.choices[0].message.content.strip()

        unwanted_phrases = [
            "Here is a concise description of the book:",
            "Here is a brief summary of the book:",
            "This is a brief description of the book:",
            "Here is a concise and clear description of the book:",
            "This book,"
        ]
        for phrase in unwanted_phrases:
            if description.startswith(phrase):
                description = description.replace(phrase, "").strip()

        return description


# Beautify and summarize search results
def beautify_results(results):
    beautified_output = []
    unique_books = {}

    for result, score in results:
        if result not in unique_books:
            unique_books[result] = {"info": result, "distance": score}


    for book in unique_books.values():
        book_info = book["info"]
        description = summarize_book(book_info)
        beautified_output.append({
            "book": book_info.split(' - ')[1],
            "author": book_info.split(' - ')[0],
            "publisher": book_info.split(' - ')[2],
            "year": book_info.split(' - ')[3],
            "description": description
        })

    return beautified_output

# Displaying Result 
def display_results(beautified_results):
    for book in beautified_results:
        st.markdown(f"""
        **Book:** {book['book']}  
        **Author:** {book['author']}  
        **Publisher:** {book['publisher']}  
        **Year:** {book['year']}  
        **Description:** {book['description']}  
        """, unsafe_allow_html=True)


# Data creation for audio
def summarize_results_for_audio(beautified_results):

    output = []
    for idx, book in enumerate(beautified_results, start=1):
        txt=book['author'].split(", ")
        author = txt[1] + txt[0]
#         print(author)
        output.append(f"{book['book']} by {author}, published by {book['publisher']} in year {book['year']}")

        # print(output)

    return "Hello Everyone, I got these results:\n" + "\n".join(output)


#Generating audio with text
def generate_audio(summary_text):
    command = [
        "python", "python-clients/scripts/tts/talk.py",
        "--server", "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata", "function-id", "0149dedb-2be8-4195-b9a0-e57e0e14f972",
        "--metadata", "authorization", "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",  # Replace with your API key
        "--text", summary_text,
        "--voice", "English-US.Female-1",
        "--output", "audio.wav"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # print("Audio generation successful:", result.stdout)
        return "audio.wav"  # Return the audio file path
    except subprocess.CalledProcessError as e:
        print("Error occurred while generating audio:", e.stderr)
        return None
    
# Autoplay audio
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )



# Check for existing FAISS database
def load_faiss_database():
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        loaded_index = faiss.read_index(index_file)
        loaded_metadata = np.load(metadata_file, allow_pickle=True)
        return loaded_index, loaded_metadata
    return None, None

index, metadata = load_faiss_database()

# Layout Configuration
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 3])

# Left Side: Upload and Create Database
with col1:
    st.header("Upload CSV File")
    
    if index is not None:
        st.success("Database loaded successfully!")
    else:
        st.warning("No Database found. Please upload a CSV file to create one.")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        st.write("Processing uploaded file...")
        index, metadata = create_faiss_database(uploaded_file)
        st.success("Database created successfully!")

# Right Side: Query Search
with col2:
    st.image("logo2.png", width=500)
    st.header("Search Your Book")
    
    if index is not None:
        query = st.text_input("Enter query")
        
        if query:
            results = search_faiss(query)
            beautified_results = beautify_results(results)
            summary_text=summarize_results_for_audio(beautified_results)
            audio_file = generate_audio(summary_text)
            display_results(beautified_results)
            # Generate audio from the summary text

            autoplay_audio(audio_file)
            
            if audio_file:
                st.audio(audio_file, format="audio/wav")
    else:
        st.warning("Please upload a CSV file to create a Database first.")