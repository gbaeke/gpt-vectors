import os
import pinecone
import openai
import tiktoken

# use cl100k_base tokenizer for gpt-3.5-turbo and gpt-4
tokenizer = tiktoken.get_encoding('cl100k_base')


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# get the Pinecone API key and environment
pinecone_api = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(api_key=pinecone_api, environment=pinecone_env)

# set index
index = pinecone.Index('blog-index')

while True:
    # set query
    your_query = input("\nWhat would you like to know? ")
    
    # vectorize your query with openai
    try:
        query_vector = openai.Embedding.create(
            input=your_query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        print("Error calling OpenAI Embedding API: ", e)
        continue

    # search for the most similar vector in Pinecone
    search_response = index.query(
        top_k=5,
        vector=query_vector,
        include_metadata=True)

    # create a list of urls from search_response['matches']['metadata']['url']
    urls = [item["metadata"]['url'] for item in search_response['matches']]

    # make urls unique
    urls = list(set(urls))

    # create a list of texts from search_response['matches']['metadata']['text']
    chunks = [item["metadata"]['text'] for item in search_response['matches']]

    # combine texts into one string to insert in prompt
    all_chunks = "\n".join(chunks)

    # print urls of the chunks
    print("URLs:\n\n", urls)

    # print the text number and first 50 characters of each text
    print("\nChunks:\n")
    for i, t in enumerate(chunks):
        print(f"\nChunk {i}: {t[:50]}...")

    try:
        # openai chatgpt with article as context
        # chat api is cheaper than gpt: 0.002 / 1000 tokens
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content":  "You are a thruthful assistant!" },
                { "role": "user", "content": f"""Answer the following query based on the context below ---: {your_query}
                                                    Do not answer beyond this context!
                                                    ---
                                                    {all_chunks}""" }
            ],
            temperature=0,
            max_tokens=750
        )

        print(f"\n{response.choices[0]['message']['content']}")
    except Exception as e:
        print(f"Error with OpenAI Completion: {e}")




