"""
Configuration file for Sustainability Advisor
"""

import os

MODEL_DIR = "models"
DATA_DIR = "data"

MODEL_PATH = os.path.join(MODEL_DIR, "co2_prediction_model.pkl")
ACTIVITY_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_activity.pkl")
CATEGORY_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_category.pkl")
MAPPING_PATH = os.path.join(MODEL_DIR, "activity_category_mapping.json")
TIPS_PATH = os.path.join(DATA_DIR, "sustainability_tips.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_COLLECTION_NAME = "sustainability_tips"
CHROMA_PERSIST_DIR = ".chroma"

PAGE_TITLE = "🌍 Sustainability Advisor"
PAGE_ICON = "🌱"
LAYOUT = "wide"

HIGH_EMISSION_THRESHOLD = 4.0
MEDIUM_EMISSION_THRESHOLD = 1.0

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

COLORS = {
    "primary": "#4CAF50",
    "secondary": "#2E7D32",
    "accent": "#8BC34A",
    "background": "#E8F5E9",
    "warning": "#FF6B6B",
    "info": "#90CAF9"
}
