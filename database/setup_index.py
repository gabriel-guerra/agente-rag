from pymilvus import connections, utility, Collection
import pandas as pd
from google import genai

# Client Gemini
client = genai.Client()

# Connect to Milvus Database
connections.connect(host='localhost', port='19530')

# Access collection created on ./create_database.py
collection = Collection("gemini_collection")

# Start database data
content = "Hello world!"

# Generate Embedding   
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=content
)

# Structure in a pandas dataframe
data = pd.DataFrame({
    "embedding": [result.embeddings[0].values],
    "text": content
})

# Insert into Database
collection.insert(data)

# Create index to habilitate collection for searches 
collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)