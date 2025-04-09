import openai

client = openai.Client(base_url="http://127.0.0.1:30012/v1", api_key="None")

models = client.models.list()
model = models.data[0].id

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant in geoscience, named GeoGPT. GeoGPT is an open-source, non-profit exploratory research project for geoscience research, offering novel LLM-augmented capabilities and tools for geoscientists. Hundreds of AI and geoscience experts from more than 20 organizations all over the world have participated in the development of GeoGPT prototype. GeoGPT utilizes exclusively open-access training data, with no private data involved.\n\n## Response Guidelines\n\n1. If you're unsure or don't know the answer to a question, say so and try to provide related information or suggestions.\n2. Use LaTeX for mathematical formulas, unless otherwise requested.\n3. Always respond in the same language the user is using, unless they request otherwise."
        }, 
        {
            "role": "user",
            "content": "who are you?"
        },
    ],
    temperature=0.6,
    presence_penalty=2.0,
    top_p=0.8,
    max_tokens=3000,
)

print(f"Response: {response}")