# 🌍 Sustainability Advisor - AI-Powered Carbon Footprint Reduction

An intelligent sustainability recommendation system that predicts CO2 emissions and provides personalized tips using Machine Learning, Vector Embeddings, and Agentic AI.

## 🚀 Features

- **CO2 Emission Prediction**: Random Forest model trained on activity data
- **Vector Store**: ChromaDB with sentence transformers for semantic search
- **Agentic AI**: LangChain-powered intelligent agent workflow
- **Interactive UI**: Beautiful Streamlit interface with visualizations
- **Dataset Analysis**: Bulk processing and comprehensive analytics

## 🛠️ Tech Stack

- **ML Model**: Random Forest Regressor (scikit-learn)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Agent Framework**: LangChain
- **UI**: Streamlit
- **Visualization**: Plotly

## 📦 Installation & Setup

### Step 1: Train the Model (Google Colab)

1. Open `CO2_Model_Training.ipynb` in Google Colab
2. Upload your `activity_co2_emission_data.csv`
3. Run all cells sequentially
4. Download these files:
   - `co2_prediction_model.pkl`
   - `label_encoder_activity.pkl`
   - `label_encoder_category.pkl`
   - `activity_category_mapping.json`
   - `sustainability_tips.json`

### Step 2: Setup Local Project

```bash
mkdir sustainability-advisor
cd sustainability-advisor
mkdir models data

# Place downloaded files in appropriate folders
# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🎯 Usage

### Single Activity Analysis
1. Navigate to "Single Activity Analysis"
2. Select activity (Car, Bus, Bicycle, AC)
3. Select category (Transport, Household)
4. Click "Analyze Activity"
5. Get CO2 prediction and personalized recommendations

### Dataset Analysis
1. Navigate to "Dataset Analysis"
2. Upload your CSV file
3. View statistics and visualizations
4. Click "Get AI Recommendations"

## 📁 Project Structure

```
sustainability-advisor/
├── app.py
├── agent.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── co2_prediction_model.pkl
│   ├── label_encoder_activity.pkl
│   ├── label_encoder_category.pkl
│   └── activity_category_mapping.json
└── data/
    ├── sustainability_tips.json
    └── activity_co2_emission_data.csv
```

## 🐛 Troubleshooting

**Error: Model files not found**
- Ensure .pkl files are in `models/` directory

**Error: ChromaDB initialization failed**
- Run: `pip install chromadb --upgrade`

**Streamlit not loading**
- Run: `streamlit run app.py --server.port 8502`

## 📄 License

Open source - MIT License

---
Made with 💚 for a sustainable future 🌍
