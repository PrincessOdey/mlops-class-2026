# BonsAI Pulse

BonsAI Pulse helps bonsai owners at every stage: diagnosing a sick tree, keeping a
healthy one on a care rhythm, and guiding total beginners before they even buy one.

## How it works

One entry point, three outcomes, powered by the same trained classifier and the same
Gemini prompt engine:

- **Rescue** — describe your tree's symptoms, get a diagnosis (via a trained ML model)
  and a step-by-step fix.
- **Care Calendar** — if the model says your tree is healthy, get an ongoing weekly
  care rhythm instead.
- **Learn Before Buying** — new to bonsai? Get a beginner's guide for a species you're
  considering, no tree (or symptoms) required.

## Architecture

| Component | What it does |
|---|---|
| `01_generate_dataset.ipynb` | Generates a synthetic dataset of bonsai symptoms and their likely cause |
| `02_train_model.ipynb` | Trains a decision tree classifier, logs runs to MLflow, registers the best one as `champion` |
| `03_prompts.ipynb` | Designs and logs two Gemini prompt versions to MLflow |
| `api/main.py` | FastAPI service: loads the champion model from MLflow's registry, calls Gemini, exposes `/diagnose` and `/learn` |
| `ui/index.html` | Simple browser UI hitting the API |
| `docker/docker-compose.yml` | Runs MLflow, JupyterLab, and the API together |

## Running it

1. Copy `.env.example` to `.env` and add your own `GOOGLE_API_KEY`
   (get one free at https://aistudio.google.com/apikey).
2. From `docker/`, run: