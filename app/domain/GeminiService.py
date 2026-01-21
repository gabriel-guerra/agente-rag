from pymilvus import connections, utility, Collection
import pandas as pd
from google import genai


class GeminiService:
    
    def __init__(self):
        # Client Gemini
        self.client = genai.Client()
        
        # Access database collection
        self.collection = Collection("gemini_collection")
        self.collection.load()


    def generate_embedding(self, prompt):
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=prompt
        )
        return [result.embeddings[0].values]

    def save_embedding(self, prompt):
        data = pd.DataFrame({
            "embedding": self.generate_embedding(prompt),
            "text": prompt
        })
        self.collection.insert(data)

    def search_context(self, prompt):    
        search = self.collection.search(
            data=self.generate_embedding(prompt),
            anns_field='embedding',
            param={
                "metric_type": "COSINE",
                "params": {
                    "nprobe": 16,
                }
            },
            limit=1,
            output_fields=["id", 'text']
        )
        return search[0][0]['entity']['text']
    
    def generate(self, prompt):
        context = self.search_context(prompt)

        if context: 
            content = f"{prompt}\n\n Context: {context}"
        else: 
            content = prompt
        
        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=content
        )

        self.save_embedding(response.text)
        return response.text

