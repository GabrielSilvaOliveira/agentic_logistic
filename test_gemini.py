from llm.gemini_provider import GeminiProvider

llm = GeminiProvider()

response = llm.generate(
    prompt="Explain what a supply chain delay is in logistics."
)

print(response)