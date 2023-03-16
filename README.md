For more information see this [blog post](https://blog.baeke.info/2023/03/16/pinecone-and-openai-magic-a-guide-to-finding-your-long-lost-blog-posts-with-vectorized-search-and-chatgpt/)

To run the sample web app:

- Get an account at Pinecone and create an index as explained in the blog post
- Get an account at OpenAI
- Set environment variables for Pinecone and your OpenAI key (see blog post)
- From the `console` folder, use upload_vectors.py to upload blog posts as vectors to Pinecone. You can use another feed if you like.
- From the `webapp` folder, run `app.py` (e.g. python3 app.py). This will start a web server on port 5000 allowing you to search for blog posts.

