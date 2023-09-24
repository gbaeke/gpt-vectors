import sys
import pathlib

# add helpers folder to path (required for Streamlit to find the helpers module)
sys.path.append(str(pathlib.Path().absolute()) + "/helpers")


import feedparser
import os
import openai
import requests
from bs4 import BeautifulSoup
from retrying import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import PGVector
import tiktoken
import hashlib
import streamlit as st
import urllib.parse
from helpers import tiktoken_len, crawl

import dotenv

dotenv.load_dotenv(dotenv_path='../../.env')

# check environment variables
if os.getenv('OPENAI_API_KEY') is None:
    st.error("OPENAI_API_KEY not set. Please set this environment variable and restart the app.")
    st.stop()
if os.getenv("CONNECTION_STRING") is None:
    st.error("CONNECTION_STRING not set. Please set this environment variable and restart the app.")
    st.stop()

# app starts here
st.title("Upload content to vector db 🔎")

st.write("Click Upload to store contect in vector db")

url = st.text_input("Address", "https://blog.baeke.info/feed/")

# create unique hash for url
url_hash = hashlib.sha256(url.encode()).hexdigest()

address_type = st.selectbox("Address type", ["RSS", "Crawl"])

urls = []
if address_type == "RSS":
    # Parse the RSS feed with feedparser
    st.write("Parsing RSS feed: ", url)

    try:
        urls = feedparser.parse(url)
    except Exception as e:
        st.exception(e)
        st.stop()

    # store all the entries in a pages list and display
    # number of pages
    pages = urls.entries
    num_pages = len(pages)
    if num_pages == 0:
        st.write("Error processing feed. Stopping...")
        st.stop()
    st.write("Number of entries: ", num_pages)
elif address_type == "Crawl":
    # fill entries with all links until a certain depth
    pages = crawl(url, 1)
    num_pages = len(pages)
    if num_pages == 0:
        st.write("Error processing feed. Stopping...")
        st.stop()
    st.write("Number of entries: ", num_pages)
    print(pages)

with st.expander("Options", expanded=False):
    chunk_size = st.slider("Chunk size", 100, 600, 400)
    chunk_overlap = st.slider("Chunk overlap", 0, 60, 20)
    blog_entries = st.slider("Blog entries", 1, num_pages, num_pages)
    overwrite = st.checkbox("Overwrite existing entries", False)

if st.button("Upload"):
    # OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    connection_string = os.getenv("CONNECTION_STRING")
    collection_name = "gpt_vectors"

    # create recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,  # number of tokens overlap between chunks
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )

    # starting the upload process
    progress_text = "Creating docs..."
    my_bar = st.progress(0, text=progress_text)

    all_docs = []

    with st.expander("Logs", expanded=False):
        for i, entry in enumerate(pages[:blog_entries]):
            page = entry['link']
            r = requests.get(page)
            soup = BeautifulSoup(r.text, 'html.parser')
            if url == "https://blog.baeke.info/feed/":
                article = soup.find("div", {"class": "entry-content"}).text
            else:
                article = soup.text

            st.write("Processing URL: ", page)

            # create documents from split text
            docs = text_splitter.create_documents([article], metadatas=[{"url": page}])        
    
            all_docs.extend(docs)
           
               
            # update progress bar
            my_bar.progress((i+1)/blog_entries, text=progress_text + f" {i+1} of {blog_entries}")

        my_bar.progress(1.0, text="Docs created")

    # now we have all the chunks, we create embeddings with LangChain's pgvector
    with st.spinner("Creating embeddings..."):
        embeddings = OpenAIEmbeddings()
        db = PGVector.from_documents(
            embedding=embeddings,
            documents=all_docs,
            collection_name=collection_name,
            connection_string=connection_string,
            pre_delete_collection=overwrite,

        )

    
