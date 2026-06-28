import joblib
import numpy as np
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading

# ── Load model & pipeline from HuggingFace Model Hub ─────
REPO_ID = "Shaddy001/california-housing-predictor"

print("⏳ Downloading model from HuggingFace...")
model_path    = hf_hub_download(repo_id=REPO_ID, filename="model.pkl")
pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="pipeline.pkl")

model    = joblib.load(model_path)
pipeline = joblib.load(pipeline_path)
print("✅ Model loaded successfully!")

# ── FastAPI app ───────────────────────────────────────────
fastapi_app = FastAPI(
    title       = "California Housing Price Predictor",
    description = "Predicts median house value using a tuned RandomForest model.",
    version     = "1.0.0"
)

class HouseFeatures(BaseModel):
    longitude           : float
    latitude            : float
    housing_median_age  : float
    total_rooms         : float
    total_bedrooms      : float
    population          : float
    households          : float
    median_income       : float
    ocean_proximity     : str

@fastapi_app.get("/")
def root():
    return {"message": "California Housing Price Predictor API is running!"}

@fastapi_app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@fastapi_app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.model_dump()])
    transformed = pipeline.transform(input_df)
    prediction  = model.predict(transformed)[0]
    prediction  = float(np.clip(prediction, 14999, 500001))
    return {"predicted_house_value": round(prediction, 2)}

# ── Gradio prediction function ────────────────────────────
OCEAN_OPTIONS = ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"]

def predict_gradio(longitude, latitude, housing_median_age,
                   total_rooms, total_bedrooms, population,
                   households, median_income, ocean_proximity):
    try:
        input_df = pd.DataFrame([{
            "longitude"          : longitude,
            "latitude"           : latitude,
            "housing_median_age" : housing_median_age,
            "total_rooms"        : total_rooms,
            "total_bedrooms"     : total_bedrooms,
            "population"         : population,
            "households"         : households,
            "median_income"      : median_income,
            "ocean_proximity"    : ocean_proximity,
        }])

        transformed = pipeline.transform(input_df)
        prediction  = model.predict(transformed)[0]
        prediction  = float(np.clip(prediction, 14999, 500001))

        return f"🏠 Predicted House Value: ${prediction:,.2f}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ── CSS ───────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #13131a;
    --card:    #1a1a24;
    --accent:  #4f8ef7;
    --accent2: #4ecdc4;
    --text:    #f0eff8;
    --muted:   #8887a0;
    --border:  #2a2a3a;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

#header { text-align: center; padding: 2rem 1rem 1rem; }
#logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    letter-spacing: -1px; color: var(--text);
}
#logo span { color: var(--accent); }
#tagline {
    display: inline-block; margin-top: 6px;
    background: #1a1e3a; border: 1px solid var(--accent);
    color: var(--accent); font-size: 11px;
    padding: 3px 14px; border-radius: 20px;
    letter-spacing: 1.5px; font-weight: 500;
}

label span {
    font-size: 12px !important; font-weight: 500 !important;
    color: var(--muted) !important; letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}

input[type="number"], input[type="text"], select, textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

#predict-btn {
    background: var(--accent) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-family: 'Syne', sans-serif !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 13px !important; width: 100% !important;
    transition: opacity 0.2s !important;
}
#predict-btn:hover { opacity: 0.85 !important; }

#result-box {
    background: #0f1a2a !important;
    border: 1px solid var(--accent) !important;
    border-radius: 12px !important;
    color: var(--accent) !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    text-align: center !important;
    padding: 1rem !important;
}

.gr-box, .gr-form, .block {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

footer { display: none !important; }
"""

# ── Gradio UI ─────────────────────────────────────────────
with gr.Blocks(css=css, theme=gr.themes.Base(), title="🏠 California Housing Predictor") as demo:

    gr.HTML("""
    <div id="header">
        <div id="logo">🏠 California <span>Housing</span> Predictor</div>
        <div id="tagline">ML POWERED &nbsp;·&nbsp; RANDOM FOREST &nbsp;·&nbsp; R² = 0.97</div>
    </div>
    """)

    with gr.Row():
        # Left column — Location & House Info
        with gr.Column():
            gr.Markdown("### 📍 Location")
            longitude = gr.Number(label="Longitude", value=-122.23)
            latitude  = gr.Number(label="Latitude",  value=37.88)
            ocean_proximity = gr.Dropdown(
                choices=OCEAN_OPTIONS,
                value="NEAR BAY",
                label="Ocean Proximity"
            )

            gr.Markdown("### 🏡 House Info")
            housing_median_age = gr.Slider(1, 52, value=41, step=1, label="Housing Median Age")
            median_income      = gr.Slider(0.5, 15.0, value=8.3, step=0.1, label="Median Income (in $10k)")

        # Right column — Population & Rooms
        with gr.Column():
            gr.Markdown("### 🏘️ Block Stats")
            total_rooms    = gr.Number(label="Total Rooms",    value=880)
            total_bedrooms = gr.Number(label="Total Bedrooms", value=129)
            population     = gr.Number(label="Population",     value=322)
            households     = gr.Number(label="Households",     value=126)

            gr.Markdown("### ")
            predict_btn = gr.Button("🔍 Predict House Price", elem_id="predict-btn", variant="primary")
            result      = gr.Textbox(label="💰 Prediction Result", elem_id="result-box", interactive=False)

    predict_btn.click(
        fn=predict_gradio,
        inputs=[longitude, latitude, housing_median_age,
                total_rooms, total_bedrooms, population,
                households, median_income, ocean_proximity],
        outputs=result
    )

    gr.HTML("""
    <div style="text-align:center;margin-top:2rem;font-size:11px;color:#8887a0;letter-spacing:1px;">
        CALIFORNIA HOUSING PREDICTOR &nbsp;·&nbsp; RANDOMFOREST &nbsp;·&nbsp; BY ABU SHADAB KHAN
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
