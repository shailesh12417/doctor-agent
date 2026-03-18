from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize Groq model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0   # you can also use mixtral-8x7b
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["symptoms"],
    template="""
You are a professional medical assistant.

Rules:
- Only answer health-related questions
- If not health-related, reply:
  "Sorry, I only answer health-related questions."
- Do NOT give exact diagnosis
- Keep answers simple and clear

Patient Symptoms:
{symptoms}

Provide response in this structured format:

1. Possible Causes:
- List 2–4 possible reasons

2. Severity Level:
- Mild / Moderate / Serious (choose one)
- Give a short reason

3. General Advice:
- Give 3–5 practical suggestions

4. When to See a Doctor:
- Clearly mention warning signs

5. Disclaimer:
- This is not a medical diagnosis

Answer:
"""
)

chain = prompt | llm


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, symptoms: str = Form(...)):
    response = chain.invoke({"symptoms": symptoms})
    answer = response.content

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": answer,
            "symptoms": symptoms
        }
    )