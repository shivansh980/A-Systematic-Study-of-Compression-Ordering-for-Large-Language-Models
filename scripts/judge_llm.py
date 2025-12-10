from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel
import google.generativeai as genai
import instructor
from deepeval.models import DeepEvalBaseLLM
import time
from typing import Optional

genai.configure()

class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self, requests_per_minute: int = 10):
        self.model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")
        self.requests_per_minute = requests_per_minute
        self.request_interval = 60.0 / requests_per_minute  # Time between requests in seconds
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't exceed the rate limit"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            sleep_time = self.request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def load_model(self):
        return self.model
    
    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        self._rate_limit()  # Apply rate limiting before each request
        
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp
    
    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)
    
    def get_model_name(self):
        return "Gemini 2.5 Flash"

