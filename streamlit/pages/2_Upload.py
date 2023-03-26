import feedparser
import os
import pinecone
import openai
import requests
from bs4 import BeautifulSoup
from retrying import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import hashlib
import streamlit as st

# check environment variables
if os.getenv('PINECONE_API_KEY') is None:
    st.stop("PINECONE_API_KEY not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_ENVIRONMENT') is None:
    st.stop("PINECONE_ENVIRONMENT not set. Please set this environment variable and restart the app.")
if os.getenv('OPENAI_API_KEY') is None:
    st.stop("OPENAI_API_KEY not set. Please set this environment variable and restart the app.")

# use cl100k_base tokenizer for gpt-3.5-turbo and gpt-4
tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function used by the RecursiveCharacterTextSplitter
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def create_embedding(article):
    # vectorize with OpenAI text-emebdding-ada-002
    embedding = openai.Embedding.create(
        input=article,
        model="text-embedding-ada-002"
    )

    return embedding["data"][0]["embedding"]

st.title("Upload blog feed to Pinecone 🔎")

st.write("Click Upload to upload baeke.info posts to Pinecone in chunks.")

url = st.text_input("RSS feed", "https://blog.baeke.info/feed/")

# Parse the RSS feed with feedparser
st.write("Parsing RSS feed: ", url)

try:
    feed = feedparser.parse(url)
except Exception as e:
    st.exception(e)
    st.stop()

# get number of entries in feed
entries = len(feed.entries)
if entries == 0:
    st.write("Error processing feed. Stopping...")
    st.stop()
st.write("Number of entries: ", entries)

with st.expander("Options", expanded=False):
    chunk_size = st.slider("Chunk size", 100, 600, 400)
    chunk_overlap = st.slider("Chunk overlap", 0, 60, 20)
    blog_entries = st.slider("Blog entries", 1, entries, entries)
    recreate = st.checkbox("Recreate index", True)

if st.button("Upload"):
    # OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # get the Pinecone API key and environment
    pinecone_api = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

    pinecone.init(api_key=pinecone_api, environment=pinecone_env)

    if "blog-index" not in pinecone.list_indexes():
        st.write("Index does not exist. Creating...")
        pinecone.create_index("blog-index", 1536, metadata_config= {"indexed": ["url", "chunk-id"]})
    else:
        st.write("Index already exists.")
        if recreate:
            st.write("Deleting existing index...")
            pinecone.delete_index("blog-index")
            st.write("Creating new index...")
            pinecone.create_index("blog-index", 1536, metadata_config= {"indexed": ["url", "chunk-id"]})
        else:
            st.write("Reusing existing index.")

    # set index; must exist
    index = pinecone.Index('blog-index')

    # create recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,  # number of tokens overlap between chunks
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )

    # starting the upload process
    progress_text = "Upload in progress..."
    my_bar = st.progress(0, text=progress_text)

    pinecone_vectors = []

    with st.expander("Logs", expanded=False):
        for i, entry in enumerate(feed.entries[:50]):
            r = requests.get(entry.link)
            soup = BeautifulSoup(r.text, 'html.parser')
            article = soup.text

            st.write("Processing URL: ", entry.link)

            # create chunks
            chunks = text_splitter.split_text(article)

            # create md5 hash of entry.link
            url = entry.link
            url_hash = hashlib.md5(url.encode("utf-8"))       
            url_hash = url_hash.hexdigest()

            number_of_chunks = len(chunks)

            # create embeddings for each chunk
            for j, chunk in enumerate(chunks):
                st.write("\tCreating embedding for chunk ", j+1, " of ", number_of_chunks)
                vector = create_embedding(chunk)

                # concatenate hash and j
                hash_j = url_hash + str(j)

                # add vector to pinecone_vectors list
                st.write("\tAdding vector to pinecone_vectors list for chunk ", j+1, " of ", number_of_chunks)
                pinecone_vectors.append((hash_j, vector, {"url": entry.link, "chunk-id": j, "text": chunk}))

                # upsert every 100 vectors
                if len(pinecone_vectors) % 100 == 0:
                    st.write("Upserting batch of 100 vectors...")
                    upsert_response = index.upsert(vectors=pinecone_vectors)
                    pinecone_vectors = []
            
            # update progress bar
            my_bar.progress((i+1)/entries, text=progress_text + f" {i+1} of {entries}")

    # if there are any vectors left, upsert them
    if len(pinecone_vectors) > 0:
        upsert_response = index.upsert(vectors=pinecone_vectors)
        pinecone_vectors = []

    my_bar.progress(1.0, text="Upload complete.")
