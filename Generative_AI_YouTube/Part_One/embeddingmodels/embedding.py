from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACEHUB_API_KEY")
)

text = input("Enter text: ")

embedding = client.feature_extraction(
    text,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print(f"\nDimension: {len(embedding)}")
print("\nFirst 10 values:")
print(embedding[:10])

# cd "Part_One/embeddingmodels"
# python embedding.py