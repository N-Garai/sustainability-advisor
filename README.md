# 🌍 Sustainability Advisor

> **An AI-powered platform for predicting CO2 emissions and providing personalized sustainability recommendations using Machine Learning and Vector Embeddings.**


**🔗 Live Demo:** [Sustainability Advisor App](https://sustainability-advisorgit-zz8ek7o6pwseqj6uutnqhy.streamlit.app/#212cfb73)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🌟 Overview

**Sustainability Advisor** is an intelligent web application that helps individuals and organizations reduce their carbon footprint through:

- **CO2 Emission Predictions** using Random Forest machine learning models
- **AI-Powered Recommendations** through semantic vector search
- **Interactive Data Visualizations** with Plotly charts
- **Comprehensive Dataset Analysis** for bulk carbon footprint assessment

The app combines classical ML, embeddings-based retrieval, and modern web technologies to make sustainability accessible and actionable.

---

## ✨ Features

### 🎯 Core Functionality

| Feature | Description |
|---------|-------------|
| **Single Activity Analysis** | Predict CO2 emissions for individual activities (Car, Bus, Bicycle, AC usage) |
| **Dataset Analysis** | Upload CSV files for bulk emission analysis and insights |
| **Smart Recommendations** | Get personalized sustainability tips using vector similarity search |
| **Interactive Dashboards** | Visualize emissions data with beautiful Plotly charts |
| **Sustainability Tips Database** | Browse 18+ expert-curated tips filtered by category |

### 🤖 AI/ML Components

- **Random Forest Regressor**: Predicts CO2 emissions based on activity and category
- **Sentence Transformers**: Generates embeddings for semantic tip retrieval (`all-MiniLM-L6-v2`)
- **ChromaDB Vector Store**: Efficient similarity search for recommendations
- **Label Encoding**: Transforms categorical features for ML model input

---

## 🎥 Demo

**Try it live:** [Sustainability Advisor App](https://sustainability-advisorgit-zz8ek7o6pwseqj6uutnqhy.streamlit.app/#212cfb73)

### Quick Start:
1. Select **"Single Activity Analysis"** from the sidebar
2. Choose an activity (e.g., Car(20km)) and category (Transport)
3. Click **"Analyze Activity"** to see:
   - CO2 emission prediction
   - 3 personalized recommendations
   - Comparison charts

---

## 🛠️ Tech Stack

### **Frontend**
- **Streamlit** - Interactive web framework
- **Plotly** - Data visualization

### **Machine Learning**
- **scikit-learn** - Random Forest model
- **joblib** - Model persistence
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### **AI & Embeddings**
- **sentence-transformers** - Text embeddings (all-MiniLM-L6-v2)
- **ChromaDB** - Vector database for similarity search

### **Development**
- **Python 3.11+**
- **Git** - Version control
- **Streamlit Cloud** - Hosting

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/N-Garai/sustainability-advisor.git
cd sustainability-advisor
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Train the Model

**Option A: Use Google Colab (Recommended)**
1. Open `CO2_Model_Training.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `data/activity_co2_emission_data.csv`
3. Run all cells
4. Download generated files:
   - `co2_prediction_model.pkl`
   - `label_encoder_activity.pkl`
   - `label_encoder_category.pkl`
   - `activity_category_mapping.json`
   - `sustainability_tips.json`

**Option B: Local Training**
```bash
jupyter notebook CO2_Model_Training.ipynb
# Follow notebook instructions
```

#### 5. Organize Model Files
```
sustainability-advisor/
├── models/
│   ├── co2_prediction_model.pkl          # From Colab
│   ├── label_encoder_activity.pkl        # From Colab
│   ├── label_encoder_category.pkl        # From Colab
│   └── activity_category_mapping.json    # From Colab
└── data/
    ├── sustainability_tips.json          # From Colab
    └── activity_co2_emission_data.csv    # Provided
```

#### 6. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🚀 Usage

### 1️⃣ Single Activity Analysis
**Predict emissions for one activity:**
```
1. Navigate to "🔍 Single Activity Analysis"
2. Select activity (Car, Bus, Bicycle, AC)
3. Select category (Transport, Household)
4. Click "Analyze Activity"
5. View prediction + 3 AI recommendations
```

### 2️⃣ Dataset Analysis
**Bulk process multiple activities:**
```
1. Navigate to "📊 Dataset Analysis"
2. Upload CSV with columns: Activity, AVG CO2 emission, Category
3. View statistics and visualizations
4. Click "Get AI Recommendations" for insights
```

**Example CSV Format:**
```csv
Activity,AVG CO2 emission,Category
Car(20km),4.6,Transport
Bus(20km),1.2,Transport
Bicycle(20km),0.0,Transport
AC usage(8hrs/day),5.5,Household
```

### 3️⃣ Browse Sustainability Tips
```
1. Navigate to "💡 Sustainability Tips"
2. Filter by category (Transport, Household, General)
3. Explore 18+ expert tips with impact metrics
```

---

## 📁 Project Structure

```
sustainability-advisor/
│
├── app.py                              # Main Streamlit application
├── agent.py                            # AI agent (ML + Vector search)
├── config.py                           # Configuration settings
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System dependencies (for Streamlit Cloud)
├── README.md                           # This file
├── .gitignore                          # Git exclusions
│
├── models/                             # ML models (from training)
│   ├── co2_prediction_model.pkl
│   ├── label_encoder_activity.pkl
│   ├── label_encoder_category.pkl
│   └── activity_category_mapping.json
│
├── data/                               # Data files
│   ├── sustainability_tips.json
│   └── activity_co2_emission_data.csv
│
└── CO2_Model_Training.ipynb            # Jupyter notebook for model training
```

---

## 🧠 Model Training

### Dataset
The model is trained on `activity_co2_emission_data.csv` containing:
- **Activity**: Type of activity (e.g., Car(20km), Bus(20km))
- **AVG CO2 emission**: Carbon footprint in kg
- **Category**: Activity category (Transport, Household)

### Training Pipeline
1. **Data Preprocessing**: Label encoding for categorical features
2. **Model Selection**: Random Forest Regressor (100 estimators, max_depth=10)
3. **Vector Store Creation**: Sustainability tips embedded with Sentence Transformers
4. **Model Serialization**: Save as `.pkl` files for deployment

### Model Performance
- **Algorithm**: Random Forest with 100 trees
- **Features**: Activity (encoded), Category (encoded)
- **Target**: CO2 emission (kg)
- **Evaluation Metrics**: R² score, MSE

---

## 🚀 Deployment

### Streamlit Cloud (Current Deployment)

**Live App:** [https://sustainability-advisorgit-zz8ek7o6pwseqj6uutnqhy.streamlit.app](https://sustainability-advisorgit-zz8ek7o6pwseqj6uutnqhy.streamlit.app/#212cfb73)

#### Deploy Your Own:
1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub account
   - Select repository: `N-Garai/sustainability-advisor`
   - Branch: `main`
   - Main file: `app.py`
   - Click **Deploy**

3. **Configure Secrets (if using APIs)**
   - Go to App Settings → Secrets
   - Add environment variables

### Alternative Platforms

#### Heroku
```bash
# Install Heroku CLI
heroku login

# Create app
heroku create sustainability-advisor

# Deploy
git push heroku main
```

**Required files:**
- `Procfile`: `web: streamlit run app.py --server.port=$PORT`
- `setup.sh`: Streamlit configuration

#### Railway
1. Create `Dockerfile`
2. Push to GitHub
3. Import project on [Railway](https://railway.app)
4. Auto-deploy from GitHub

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute
1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit** changes: `git commit -m 'Add AmazingFeature'`
4. **Push** to branch: `git push origin feature/AmazingFeature`
5. Open a **Pull Request**

### Ideas for Contributions
- 🌟 Add more sustainability tips
- 📊 Improve ML model accuracy
- 🎨 Enhance UI/UX design
- 🌐 Add multilingual support
- 📱 Create mobile-responsive design
- 🔌 Integrate real-time emission APIs
- 🧪 Add unit tests
- 📚 Improve documentation

---

## 📧 Contact

**Nishant Garai**
- GitHub: [@N-Garai](https://github.com/N-Garai)
- Project Link: [https://github.com/N-Garai/sustainability-advisor](https://github.com/N-Garai/sustainability-advisor)
- Live Demo: [Sustainability Advisor App](https://sustainability-advisorgit-zz8ek7o6pwseqj6uutnqhy.streamlit.app/#212cfb73)

---

## 🙏 Acknowledgments

- **Dataset**: Custom activity-emission dataset
- **Embeddings Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Icons**: [Icons8](https://icons8.com/)
- **Hosting**: [Streamlit Cloud](https://streamlit.io/cloud)
- **Inspiration**: UN Sustainable Development Goals

---

## 🚀 Quick Commands Reference

```bash
# Clone and setup
git clone https://github.com/N-Garai/sustainability-advisor.git
cd sustainability-advisor
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Deploy to Streamlit Cloud
git push origin main
# Then deploy at share.streamlit.io
```

---

## 🔬 Technical Details

### Machine Learning Pipeline
```
Input (Activity, Category) 
    → Label Encoding 
    → Random Forest Prediction 
    → CO2 Emission Output
```

### Recommendation System
```
User Query 
    → Sentence Embedding (all-MiniLM-L6-v2) 
    → ChromaDB Vector Search 
    → Top-3 Similar Tips 
    → Formatted Recommendations
```

### Data Flow
```
User Input → Streamlit UI → Agent (agent.py) → ML Model + Vector DB → Results → Plotly Visualization
```

---



### 🌱 Made with 💚 for a Sustainable Future

**[⬆ Back to Top](#-sustainability-advisor)**

---

**Last Updated:** October 15, 2025

</div>
