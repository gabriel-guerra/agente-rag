from pydantic import BaseModel
from fastapi import APIRouter
from pymilvus import connections, utility, Collection
import pandas as pd
from google import genai
from app.domain.GeminiService import GeminiService

router = APIRouter(prefix="/gemini")

class RequestPromptModel(BaseModel):
    prompt: str

@router.post("/")
def find_context(req: RequestPromptModel):
    gemini = GeminiService()
    return gemini.generate(req.prompt)