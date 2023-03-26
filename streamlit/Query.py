import os
import pinecone
import openai
import tiktoken
import streamlit as st

# check environment variables
if os.getenv('PINECONE_API_KEY') is None:
    st.error("PINECONE_API_KEY not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_ENVIRONMENT') is None:
    st.error("PINECONE_ENVIRONMENT not set. Please set this environment variable and restart the app.")
if os.getenv('OPENAI_API_KEY') is None:
    st.error("OPENAI_API_KEY not set. Please set this environment variable and restart the app.")

# use cl100k_base tokenizer for gpt-3.5-turbo and gpt-4
tokenizer = tiktoken.get_encoding('cl100k_base')


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# create a title for the app
st.title("Search blog feed 🔎")

# create a text input for the user query
your_query = st.text_input("What would you like to know?")
model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

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
        chunks = st.slider("Number of chunks", 1, max_chunks, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

    with col2:
        reply_tokens = st.slider("Reply tokens", 750, max_reply_tokens, 750)
    

# create a submit button
if st.button("Search"):
    # get the Pinecone API key and environment
    pinecone_api = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

    pinecone.init(api_key=pinecone_api, environment=pinecone_env)

    # set index
    index = pinecone.Index('blog-index')


    # vectorize your query with openai
    try:
        query_vector = openai.Embedding.create(
            input=your_query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error calling OpenAI Embedding API: {e}")
        st.stop()

    # search for the most similar vector in Pinecone
    search_response = index.query(
        top_k=chunks,
        vector=query_vector,
        include_metadata=True)

    # create a list of urls from search_response['matches']['metadata']['url']
    urls = [item["metadata"]['url'] for item in search_response['matches']]

    # make urls unique
    urls = list(set(urls))

    # create a list of texts from search_response['matches']['metadata']['text']
    chunk_texts = [item["metadata"]['text'] for item in search_response['matches']]

    # combine texts into one string to insert in prompt
    all_chunks = "\n".join(chunk_texts)

    # show urls of the chunks
    with st.expander("URLs", expanded=True):
        for url in urls:
            st.markdown(f"* {url}")
    

    with st.expander("Chunks"):
        for i, t in enumerate(chunk_texts):
            # remove newlines from chunk
            tokens = tiktoken_len(t)
            t = t.replace("\n", " ")
            st.write("Chunk ", i, "(Tokens: ", tokens, ") - ", t[:50] + "...")
    with st.spinner("Summarizing..."):
        try:
            prompt = f"""Answer the following query based on the context below ---: {your_query}
                                                        Do not answer beyond this context!
                                                        ---
                                                        {all_chunks}"""


            # openai chatgpt with article as context
            # chat api is cheaper than gpt: 0.002 / 1000 tokens
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    { "role": "system", "content":  "You are a truthful assistant!" },
                    { "role": "user", "content": prompt }
                ],
                temperature=temperature,
                max_tokens=max_reply_tokens
            )

            st.markdown("### Answer:")
            st.write(response.choices[0]['message']['content'])

            with st.expander("More information"):
                st.write("Query: ", your_query)
                st.write("Full Response: ", response)

            with st.expander("Full Prompt"):
                st.write(prompt)

            st.balloons()
        except Exception as e:
            st.error(f"Error with OpenAI Completion: {e}")
