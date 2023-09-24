import sys
import pathlib

# add helpers folder to path (required for Streamlit to find the helpers module)
sys.path.append(str(pathlib.Path().absolute()) + "/helpers")

import os
import openai
import tiktoken
import streamlit as st
from helpers import tiktoken_len, gpt
from langchain.vectorstores import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings

import dotenv

dotenv.load_dotenv(dotenv_path='../.env')

# check environment variables
if os.getenv('OPENAI_API_KEY') is None:
    st.error("OPENAI_API_KEY not set. Please set this environment variable and restart the app.")
    st.stop()
if os.getenv("CONNECTION_STRING") is None:
    st.error("CONNECTION_STRING not set. Please set this environment variable and restart the app.")
    st.stop()

# create a title for the app
st.title("Search 🔎")

# create a text input for the user query
your_query = st.text_area("What would you like to know?")
model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

# show query options to users
with st.expander("Options"):
    max_chunks = 5
    if model == "gpt-4":
        max_chunks = 15

    max_reply_tokens = 1250
    if model == "gpt-4":
        max_reply_tokens = 2000

    col1, col2 = st.columns(2)

    # model dropdown
    with col1:
        num_chunks = st.slider("Number of chunks", 1, max_chunks, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

    with col2:
        reply_tokens = st.slider("Reply tokens", 750, max_reply_tokens, 750)
    

# create a submit button
if st.button("Search"):
    # connection string to vector db
    connection_string = os.getenv("CONNECTION_STRING")

    # embeddings
    embeddings = OpenAIEmbeddings()

    # perform pgvector search and return the urls and chunks
    db = PGVector(
        collection_name="gpt_vectors",
        connection_string=connection_string,
        embedding_function=embeddings,
           
    )
    
    docs_with_score = db.similarity_search_with_score(your_query, k = num_chunks)


    # show urls of the chunks in expanded section
    with st.expander("URLs", expanded=True):
        urls = [doc[0].metadata['url'] for doc in docs_with_score]

        # Make URLs unique
        unique_urls = list(set(urls))
 
        for url in unique_urls:
            st.markdown(f"* {url}")    

    # show the chunks in collapsed section
    with st.expander("Chunks"):
        chunk_texts = [doc[0].page_content for doc in docs_with_score]
        for i, t in enumerate(chunk_texts):
            # remove newlines from chunk
            tokens = tiktoken_len(t)
            t = t.replace("\n", " ")
            st.write("Chunk ", i, "(Tokens: ", tokens, ") - ", t[:50] + "...")


    # chatgpt with article as context
    with st.spinner("Summarizing..."):
    
        open_ai_key = os.getenv('OPENAI_API_KEY')
        prompt = f"""Answer the following query based on the context below ---: {your_query}
                                                    Do not answer beyond this context!
                                                    ---
                                                    {chunk_texts}"""

        response_text, full_response = gpt(prompt, model, temperature, reply_tokens, api_key=open_ai_key)

        # if full_response is None then stop
        if full_response is None:
            st.stop()
        else:
            st.markdown("### Answer:")
            st.write(response_text)

            with st.expander("More information"):
                st.write("Query: ", your_query)
                st.write("Full Response: ", full_response)

            with st.expander("Full Prompt"):
                st.write(prompt)

            st.balloons()

