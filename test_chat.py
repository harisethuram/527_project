from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-4.1-mini",
    input="hello"
)

print(resp.output_text)