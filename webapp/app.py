import os
import logging
from flask import Flask, render_template, request, jsonify
import pinecone
import openai
import requests
from bs4 import BeautifulSoup
import tiktoken

app = Flask(__name__)

def get_highest_score_url(items):
    highest_score_item = max(items, key=lambda item: item["score"])

    if highest_score_item["score"] > 0.8:
        return highest_score_item["metadata"]['url']
    else:
        return ""

pinecone_api = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(api_key=pinecone_api, environment=pinecone_env)

index = pinecone.Index('blog-index')

openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    your_query = request.form.get('query')

    try:
        query_vector = openai.Embedding.create(
            input=your_query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        logging.error("Error calling OpenAI Embedding API: ", exc_info=True)

    search_response = index.query(
        top_k=5,
        vector=query_vector,
        include_metadata=True)
    
    print(search_response)

    url = get_highest_score_url(search_response['matches'])

    if url == "":
        return jsonify({
            'url': "",
            'response': "Only found low scoring results. Please try a different query."
        })

    logging.debug("Highest score url: %s", url)

    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    article = soup.find('div', {'class': 'entry-content'}).text

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content": "You are a polite assistant" },
                { "role": "user", "content": "Based on the article below, answer the following question: " + your_query +
                    "\nAnswer as follows:" +
                    "\nHere is the answer directly from the article:" +
                    "\nHere is the answer from other sources:" +
                     "\n---\n" + article }
                   
            ],
            temperature=0,
            max_tokens=200
        )

        response_text=f"\n{response.choices[0]['message']['content']}"
    except Exception as e:
        logging.error(f"Error with OpenAI Completion: {e}", exc_info=True)

    return jsonify({
        'url': url,
        'response': response_text
    })

if __name__ == '__main__':
    log_level = os.getenv('LOG_LEVEL', 'ERROR').upper()
    logging.basicConfig(level=getattr(logging, log_level))

    app.run(debug=True)
