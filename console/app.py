import feedparser
from sklearn.feature_extraction.text import CountVectorizer
import tiktoken
import openai
import os


# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# URL of the RSS feed to parse
url = 'https://blog.baeke.info/feed/'

# Parse the RSS feed with feedparser
feed = feedparser.parse(url)

post_texts = []

# get number of entries in feed
entries = len(feed.entries)
print("Number of entries: ", entries)

for entry in feed.entries[:entries]:
    post_texts.append(entry.title + ' ' + entry.description)

# Vectorize the post texts using scikit-learn's CountVectorizer
vectorizer = CountVectorizer()
post_vectors = vectorizer.fit_transform(post_texts)

# print vector dimensions


def tokens_from_string(string, encoding_name):
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


while True:
    your_query = input("\nWhat would you like to know? ")

    if your_query.lower() == "end":
         print("Thanks for using the app!")
         exit()

    # vectorize your query
    query_vector = vectorizer.transform([your_query])

    # calculate the similarity between your query and the blog posts
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_vector, post_vectors).flatten()

    # get the index of the most similar blog post
    most_similar_index = similarities.argmax()

    # print the most similar blog post
    print(feed.entries[most_similar_index].link)

    # scrape the content of the article
    import requests
    from bs4 import BeautifulSoup

    # get the content of the article
    r = requests.get(feed.entries[most_similar_index].link)
    soup = BeautifulSoup(r.text, 'html.parser')

    # print the content of the article
    article = soup.find('div', {'class': 'entry-content'}).text

    prompt=f'''{your_query}

    Use the context below to answer the question above.

    Context: 

    {article}
    '''


    # print number of tokens from the article with encoding cl100k_base
    num_tokens = tokens_from_string(prompt, 'cl100k_base')
    print(f"Number of tokens: {num_tokens}")

    reply_tokens = 200

    if (num_tokens + reply_tokens) > 4000:
        print('The article is too long for OpenAI to process. Please try again with a different article.')
        continue


    try:
        # openai completion with article as context
        response = openai.Completion.create(
            model="text-davinci-004",
            prompt=prompt,
            temperature=0,
            max_tokens=reply_tokens
        )

        print(response.choices[0].text)
    except Exception as e:
        print(e)

