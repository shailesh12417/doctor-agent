from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class SymptomRequest(BaseModel):
    symptoms: str

# ✅ Fix 3: Use a valid Groq model name
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0
)

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(data: SymptomRequest):
    # ✅ Fix 1 & 2: Use plain string (no f-string), reference {symptoms} via template variable
    # ✅ JSON braces escaped as {{ }} so LangChain doesn't treat them as variables
    prompt = PromptTemplate(
        input_variables=["symptoms"],
        template="""
You are a medical assistant.
Analyze the symptoms and respond ONLY in structured JSON format.

Rules:
- Only answer health-related queries
- If not health-related, return:
  {{"error": "Sorry, I only answer health-related questions."}}

For valid symptoms:
- Identify 2-3 possible conditions
- For each condition give:
  - name
  - likelihood percentage (rough estimate)
  - risk level (Low / Medium / High)
  - short explanation
- Give general advice
- Add warning if serious

Symptoms: {symptoms}

Output format (JSON only, no extra text):
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

    # ✅ Strip markdown code fences if model wraps JSON in ```json ... ```
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "error": "Model response parsing failed",
            "raw": response.content
        }

    return result
