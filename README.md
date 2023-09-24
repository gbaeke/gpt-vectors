For more information see this [blog post](https://blog.baeke.info/2023/03/16/pinecone-and-openai-magic-a-guide-to-finding-your-long-lost-blog-posts-with-vectorized-search-and-chatgpt/). This branch contains sample code to use pfvector instead of Pinecone.

To run the sample web app with pgvector:

- Create a PostgreSQL server and enable the vector extension. I used Azure Databas for PostgreSQL flexible server, the lowest tier
- Create a database on the server
- Get an account at OpenAI
- Set environment variables for OpenAI and the connection string to PostgreSQL

```
OPENAI_API_KEY=OPENAI_KEY
CONNECTION_STRING="postgresql+psycopg2://user:password@FQDN_to_server:5432/database"

```

# Streamlit App

The Streamlit app in the `streamlit` folder is a simple demo of how to use pgvector with OpenAI's GPT model. It allows you to enter a query and then generates a response using GPT-3. The app uses LangChain PGVector to retrieve the most similar blog posts to the query. Relevant chunks from those posts are put into a prompt and set to the model which provides a response.

It has both the upload and search code in one app. You can use it as a starting point for your own app.

To run the app:

- Install Python packages with requirements.txt like so: `pip install -r requirements.txt`
- From the `streamlit` folder, run `Query.py` (e.g. streamlit run Query.py). This will start a web server allowing you to upload posts via RSS and then query their contents.

