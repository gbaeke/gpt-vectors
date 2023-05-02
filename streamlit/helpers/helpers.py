import pinecone
import openai
import tiktoken
import os
import streamlit as st
from retrying import retry
import requests
from bs4 import BeautifulSoup
import urllib.parse

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    """
    Calculates the number of tokens in the given input text using a tokenizer.

    Args:
    - text (str): the text to tokenize and count the number of tokens

    Returns:
    - The number of tokens (int) in the input text.

    Description:
    This function uses a tokenizer to encode the input text into tokens, and then returns the number of tokens in the text. The tokenizer used must have a method called "encode" that takes the input text and an optional list of special characters to disallow during tokenization. The function uses the "len" built-in function to count the number of tokens returned by the tokenizer's "encode" method.
    """
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def search_pinecone(query, chunks):
    # get the Pinecone API key and environment
    pinecone_api = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

    # inititalize and set index
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)
    index = pinecone.Index('blog-index')

    # vectorize query with openai
    try:
        query_vector = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error calling OpenAI Embedding API: {e}")
        st.stop()

    # search for the most similar vector in Pinecone
    try:
        search_response = index.query(
            top_k=chunks,
            vector=query_vector,
            include_metadata=True)
    except Exception as e:
        st.error(f"Error calling Pinecone Index API: {e}")
        st.stop()
    
    # create a unique list of urls from search_response
    urls = [item["metadata"]['url'] for item in search_response['matches']]
    urls = list(set(urls))

    # create a list of texts from search_response and join them into one string
    chunk_texts = [item["metadata"]['text'] for item in search_response['matches']]
    all_chunks = "\n".join(chunk_texts)

    return urls, chunk_texts, all_chunks


def gpt(prompt, model, temperature, max_reply_tokens):

    response_text = None
    response = None
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                { "role": "system", "content":  "You are a truthful assistant!" },
                { "role": "user", "content": prompt }
            ],
            temperature=temperature,
            max_tokens=max_reply_tokens
        )

        response_text = response.choices[0]['message']['content']
    except Exception as e:
        st.error(f"Error calling OpenAI ChatCompletion API: {e}")
    
    return response_text, response


@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def create_embedding(article):
    # vectorize with OpenAI text-emebdding-ada-002
    embedding = openai.Embedding.create(
        input=article,
        model="text-embedding-ada-002"
    )

    return embedding["data"][0]["embedding"]

# get the html from a url
def get_html(url):
    response = requests.get(url)
    return response.text

# get all links from a html page
def get_links(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    base_domain = urllib.parse.urlparse(base_url).netloc
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and not href.startswith('#'):
            parsed = urllib.parse.urlparse(href)
            if not parsed.netloc or parsed.netloc == base_domain:
                links.append(urllib.parse.urljoin(base_url, href))
    return links

# recursively crawl a website and return all pages as a list of dicts
def crawl(url, depth):
    pages = []
    if depth == 0:
        return
    html = get_html(url)
    links = get_links(html, url)
    for link in links:
        pages.append({'link': link})
        crawl(link, depth-1)

    return pages