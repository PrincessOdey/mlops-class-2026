from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import os
import pandas as pd
from google import genai

app = FastAPI(title="BonsAI Pulse API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("http://mlflow:5000")

model = mlflow.pyfunc.load_model("models:/bonsai_diagnosis_model@champion")

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
import time

def generate_with_retry(prompt, model_name="gemini-3.6-flash", max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model_name, contents=prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2)

class TreeInput(BaseModel):
    leaf_color: str
    soil_moisture_pct: float
    light_hours_per_day: float
    days_since_watered: int
    pest_sighted: int
    species: str = "bonsai"

class SpeciesInput(BaseModel):
    species: str

def build_rescue_prompt(diagnosis, species):
    return f"""You are BonsAI, a warm and reassuring plant care assistant.

A customer's {species} tree shows signs of: {diagnosis}. They may be worried about their plant.

Respond in this exact format:
- One reassuring sentence (don't worry, this is fixable)
- "Steps to fix it:" followed by exactly 3 numbered steps
- One sentence on how to prevent this in future

Keep the tone warm but professional."""

def build_calendar_prompt(species):
    return f"""You are BonsAI, a friendly plant care assistant.

A customer's {species} tree is currently healthy. They want an ongoing care routine, not a rescue plan.

Respond in this exact format:
- One encouraging sentence about keeping the tree healthy
- "Your weekly care rhythm:" followed by exactly 4 bullet points covering watering, light, feeding, and one seasonal/maintenance tip
- One sentence of general encouragement

Keep the tone warm and practical, like a friendly reminder, not an emergency response."""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/diagnose")
def diagnose(tree: TreeInput):
    input_df = pd.DataFrame([tree.model_dump(exclude={"species"})])
    diagnosis = model.predict(input_df)[0]

    if diagnosis == "healthy":
        prompt = build_calendar_prompt(tree.species)
        mode = "care_calendar"
    else:
        prompt = build_rescue_prompt(diagnosis, tree.species)
        mode = "rescue"

    response = generate_with_retry(prompt)

    return {
        "mode": mode,
        "diagnosis": diagnosis,
        "advice": response.text
    }

@app.post("/learn")
def learn(input: SpeciesInput):
    prompt = build_calendar_prompt(input.species)
    response = generate_with_retry(prompt)
    return {
        "mode": "learn_before_buying",
        "advice": response.text
    }