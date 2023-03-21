To use the Azure OpenAI APIs you can add the following snippet to your code:

```python
openai.api_type = os.getenv('OPENAI_API_TYPE', 'open_ai')
openai.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
openai.api_version = os.getenv('OPENAI_API_VERSION', '2020-11-07')
```

When you call an model, use `engine` with the name of the deployment. For example:

```python
embedding = openai.Embedding.create(
    input=article,
    engine="embedding" # this expects a deployed text-embedding-ada-002 model called "embedding"
)
```