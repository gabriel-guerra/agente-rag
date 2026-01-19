from fastapi import FastAPI
from pymilvus import connections
from app.api.routes import router

app = FastAPI()
app.include_router(router)

connections.connect(host='localhost', port='19530')


