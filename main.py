from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class SymptomRequest(BaseModel):
    symptoms: str

# ✅ Valid Groq model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(data: SymptomRequest):
    try:
        # ✅ Plain string, no f-string, JSON braces escaped as {{ }}
        prompt = PromptTemplate(
            input_variables=["symptoms"],
            template="""
You are a medical assistant.
Analyze the symptoms and respond ONLY in valid JSON format. No extra text, no markdown, no code fences.

Rules:
- Only answer health-related queries
- If not health-related, return: {{"error": "Sorry, I only answer health-related questions."}}

For valid symptoms identify 2-3 possible conditions and for each give:
- name
- probability (percentage string like "70%")
- risk (exactly one of: Low / Medium / High)
- description (one sentence)

Also give:
- advice (general recommendation string)
- warning (only if serious, otherwise omit this field)

Symptoms: {symptoms}

Respond with ONLY this JSON and nothing else:
{{
  "conditions": [
    {{
      "name": "Condition name",
      "probability": "70%",
      "risk": "Medium",
      "description": "Short reason"
    }}
  ],
  "advice": "General advice here",
  "warning": "Only if serious"
}}
"""
        )

        chain = prompt | llm
        response = chain.invoke({"symptoms": data.symptoms})

        # ✅ Strip markdown fences if model wraps response
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        return JSONResponse(status_code=200, content={
            "error": "Model returned an unreadable response. Please try again.",
            "raw": response.content if 'response' in locals() else "No response"
        })
    except Exception as e:
        # ✅ Returns 200 with error JSON instead of 500
        return JSONResponse(status_code=200, content={
            "error": f"Backend error: {str(e)}"
        })
