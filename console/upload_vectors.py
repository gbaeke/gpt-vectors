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


# OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# get the Pinecone API key and environment
pinecone_api = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(api_key=pinecone_api, environment=pinecone_env)

if "blog-index" not in pinecone.list_indexes():
    print("Index does not exist. Creating...")
    pinecone.create_index("blog-index", 1536, metadata_config= {"indexed": ["url", "chunk-id"]})
else:
    print("Index already exists. Deleting...")
    pinecone.delete_index("blog-index")
    print("Creating new index...")
    pinecone.create_index("blog-index", 1536, metadata_config= {"indexed": ["url", "chunk-id"]})

# set index; must exist
index = pinecone.Index('blog-index')

# URL of the RSS feed to parse
url = 'https://blog.baeke.info/feed/'

# Parse the RSS feed with feedparser
print("Parsing RSS feed: ", url)
feed = feedparser.parse(url)

# get number of entries in feed
entries = len(feed.entries)
print("Number of entries: ", entries)

# create recursive text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

pinecone_vectors = []
for i, entry in enumerate(feed.entries[:50]):
    # report progress
    print("Create embeddings for entry ", i, " of ", entries, " (", entry.link, ")")

    r = requests.get(entry.link)
    soup = BeautifulSoup(r.text, 'html.parser')
    article = soup.find('div', {'class': 'entry-content'}).text

    # create chunks
    chunks = text_splitter.split_text(article)

    # create md5 hash of entry.link
    url = entry.link
    url_hash = hashlib.md5(url.encode("utf-8"))
    url_hash = url_hash.hexdigest()
        
    # create embeddings for each chunk
    for j, chunk in enumerate(chunks):
        print("\tCreating embedding for chunk ", j, " of ", len(chunks))
        vector = create_embedding(chunk)

        # concatenate hash and j
        hash_j = url_hash + str(j)

        # add vector to pinecone_vectors list
        print("\tAdding vector to pinecone_vectors list for chunk ", j, " of ", len(chunks))
        pinecone_vectors.append((hash_j, vector, {"url": entry.link, "chunk-id": j, "text": chunk}))

        # upsert every 100 vectors
        if len(pinecone_vectors) % 100 == 0:
            print("Upserting batch of 100 vectors...")
            upsert_response = index.upsert(vectors=pinecone_vectors)
            pinecone_vectors = []

# if there are any vectors left, upsert them
if len(pinecone_vectors) > 0:
    print("Upserting remaining vectors...")
    upsert_response = index.upsert(vectors=pinecone_vectors)
    pinecone_vectors = []

print("Vector upload complete.")




