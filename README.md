For more information see this [blog post](https://blog.baeke.info/2023/03/16/pinecone-and-openai-magic-a-guide-to-finding-your-long-lost-blog-posts-with-vectorized-search-and-chatgpt/)

To run the sample web app:

- Get an account at Pinecone and create an index as explained in the blog post
- Get an account at OpenAI
- Set environment variables for Pinecone and your OpenAI key (see blog post)
- From the `console` folder, use upload_vectors.py to upload blog posts as vectors to Pinecone. You can use another feed if you like.
- From the `webapp` folder, run `app.py` (e.g. python3 app.py). This will start a web server on port 5000 allowing you to search for blog posts.

# Streamlit App

The Streamlit app in the `streamlit` folder is a simple demo of how to use Pinecone with OpenAI's GPT-3 model. It allows you to enter a prompt and then generates a response using GPT-3. The app uses the Pinecone Python SDK to retrieve the most similar blog posts to the prompt and then uses the Pinecone REST API to retrieve the full text of the blog posts. The full text is then used as the context for the GPT-3 model.

It has both the upload and search code in one app. You can use it as a starting point for your own app.

To run the app:

- Get an account at Pinecone and create an index as explained in the blog post
- Get an account at OpenAI
- Set environment variables for Pinecone and your OpenAI key (see blog post)
- From the `streamlit` folder, run `Query.py` (e.g. streamlit run app.py). This will start a web server allowing you to search for blog posts.
- Ensure you install streamlit with `pip install streamlit`

Note: the branch **pgvector** contains the same app but uses pgvector instead.