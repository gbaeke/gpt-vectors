import os
import logging
from re import M
from flask import Flask, render_template, request, jsonify
import pinecone
import openai
import requests
from bs4 import BeautifulSoup
import tiktoken

app = Flask(__name__)

def get_highest_score_url(items):
    highest_score_item = max(items, key=lambda item: item["score"])

    if highest_score_item["score"] > 0.7:
        return highest_score_item["metadata"]['url'], highest_score_item["score"]
    
    return "", 0

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
    model = request.form.get('model')

    # set max tokens based on model
    max_tokens = 250
    if model == "gpt-4":
        max_tokens = 1024

    # if query is empty, return empty response
    if your_query == "":
        return jsonify({
            'url': "",
            'score': 0,
            'response': "Please specify a query!"
        })

    try:
        query_vector = openai.Embedding.create(
            input=your_query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        logging.error("Error calling OpenAI Embedding API: ", exc_info=True)

    search_response = []
    search_response = index.query(
        top_k=5,
        vector=query_vector,
        include_metadata=True)
    
    print(search_response)

    url, score = get_highest_score_url(search_response['matches'])

    if url == "":
        return jsonify({
            'url': "",
            'score': 0.0,
            'response': "Only found low scoring results. Please try a different query."
        })

    logging.debug("Highest score url: %s", url)

    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        article = soup.find('div', {'class': 'entry-content'}).text
    except Exception as e:
        logging.error("Error getting article: ", exc_info=True)
        return jsonify({
            'url': "",
            'score': 0,
            'response': "Error getting article. Please try again."
        })

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                { "role": "system", "content": "You are an assistant that only provides relevant answers." },
                { "role": "user", "content": "Answer me only if the article below the --- is relevant to the question. If not relevant say so and provide an answer beyond the article. If you answer beyond the article, say so. If relevant, answer in detail and with bullet points. Here is my question: " + your_query +
                     "\n---\n" + article }
                   
            ],
            temperature=0,
            max_tokens=max_tokens
        )

        response_text=f"\n{response.choices[0]['message']['content']}"
    except Exception as e:
        logging.error(f"Error with OpenAI Completion: {e}", exc_info=True)
        return jsonify({
            'url': "",
            'score': 0.0,
            'response': "Error with OpenAI Completion. Please try a different query."
        })

    return jsonify({
        'url': url,
        'score': score,
        'response': response_text
    })

if __name__ == '__main__':
    log_level = os.getenv('LOG_LEVEL', 'ERROR').upper()
    logging.basicConfig(level=getattr(logging, log_level))

    app.run(debug=True)
