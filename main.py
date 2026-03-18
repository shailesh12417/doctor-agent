from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
class SymptomRequest(BaseModel):
    symptoms: str

# Initialize Groq model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0   # you can also use mixtral-8x7b
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_question(data: SymptomRequest):


    # Prompt Template
    prompt = PromptTemplate(
    input_variables=["symptoms"],
    template=f"""
You are a medical assistant.

Analyze the symptoms and respond ONLY in structured JSON format.

Rules:
- Only answer health-related queries
- If not health-related → return:
  {{"error": "Sorry, I only answer health-related questions."}}

For valid symptoms:
- Identify 2–3 possible conditions
- For each condition give:
  - name
  - likelihood percentage (rough estimate)
  - risk level (Low / Medium / High)
  - short explanation
- Give general advice
- Add warning if serious

Symptoms: {data.symptoms}

Output format:
{{
  "conditions": [
    {{
      "name": "Condition name",
      "probability": "70%",
      "risk": "Medium",
      "description": "Short reason"
    }}
  ],
  "advice": "General advice",
  "warning": "Only if serious"
}}
"""
)
    chain = prompt | llm

    response = chain.invoke({"symptoms": data.symptoms})

    try:
        result = json.loads(response.content)
    except:
        result = {
            "error": "Model response parsing failed",
            "raw": response.content
        }

    return result
