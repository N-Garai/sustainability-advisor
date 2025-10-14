"""
Sustainability Recommendation System - Streamlit UI
CO2 Emission Prediction and Sustainability Tips using Agentic AI
"""

import streamlit as st
import pandas as pd
import os
import json
from agent import create_agent
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="🌍 Sustainability Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 2rem;
    }
    .tip-card {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    .metric-card {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_agent():
    try:
        model_path = 'models/co2_prediction_model.pkl'
        encoders_path = 'models'
        tips_json_path = 'data/sustainability_tips.json'

        if not os.path.exists(model_path):
            st.error("❌ Model files not found! Please run the training notebook first.")
            st.info("📝 Instructions:\n1. Upload and run CO2_Model_Training.ipynb in Google Colab\n2. Download all .pkl and .json files\n3. Place them in the appropriate folders")
            return None

        agent = create_agent(model_path, encoders_path, tips_json_path)
        return agent

    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def plot_emissions(df):
    col1, col2 = st.columns(2)

    with col1:
        activity_avg = df.groupby('Activity')['AVG CO2 emission'].mean().sort_values(ascending=False)
        fig1 = px.bar(
            x=activity_avg.index,
            y=activity_avg.values,
            labels={'x': 'Activity', 'y': 'Average CO2 Emission (kg)'},
            title='Average CO2 Emissions by Activity',
            color=activity_avg.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        category_sum = df.groupby('Category')['AVG CO2 emission'].sum()
        fig2 = px.pie(
            values=category_sum.values,
            names=category_sum.index,
            title='CO2 Emissions Distribution by Category',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig2, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🌍 Sustainability Advisor</h1>', unsafe_allow_html=True)
    st.markdown("### Your Personal AI Agent for Carbon Footprint Reduction")

    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing AI Agent..."):
            st.session_state.agent = initialize_agent()
            if st.session_state.agent:
                st.session_state.initialized = True
                st.success("✅ Agent initialized successfully!")

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/tree-planting.png", width=100)
        st.title("Navigation")

        page = st.radio(
            "Choose a feature:",
            ["🏠 Home", "🔍 Single Activity Analysis", "📊 Dataset Analysis", "💡 Sustainability Tips", "ℹ️ About"]
        )

        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("CO2 Saved Today", "24.5 kg", "+5.2 kg")
        st.metric("Trees Equivalent", "1.1", "+0.2")

    if page == "🏠 Home":
        show_home()
    elif page == "🔍 Single Activity Analysis":
        show_single_analysis()
    elif page == "📊 Dataset Analysis":
        show_dataset_analysis()
    elif page == "💡 Sustainability Tips":
        show_tips()
    else:
        show_about()

def show_home():
    st.markdown('<h2 class="sub-header">Welcome to Your Sustainability Journey! 🌱</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Track")
        st.write("Monitor your daily CO2 emissions")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📉 Reduce")
        st.write("Get personalized recommendations")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🌍 Impact")
        st.write("Make a real difference")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Key Features")

    features = [
        {"icon": "🤖", "title": "AI-Powered Analysis", "desc": "Advanced machine learning models predict your carbon footprint"},
        {"icon": "📊", "title": "Dataset Processing", "desc": "Upload your activity data for comprehensive analysis"},
        {"icon": "💡", "title": "Smart Recommendations", "desc": "Personalized tips using vector search and embeddings"},
        {"icon": "📈", "title": "Visual Insights", "desc": "Interactive charts and graphs for better understanding"}
    ]

    cols = st.columns(2)
    for idx, feature in enumerate(features):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="tip-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def show_single_analysis():
    st.markdown('<h2 class="sub-header">🔍 Single Activity Analysis</h2>', unsafe_allow_html=True)

    if not st.session_state.agent:
        st.warning("⚠️ Agent not initialized. Please check the model files.")
        return

    st.write("Enter your activity details to get CO2 emission prediction and personalized recommendations.")

    col1, col2 = st.columns(2)

    with col1:
        activity = st.selectbox(
            "Select Activity:",
            ["Car(20km)", "Bus(20km)", "Bicycle(20km)", "AC usage(8hrs/day)"]
        )

    with col2:
        category = st.selectbox(
            "Select Category:",
            ["Transport", "Household"]
        )

    if st.button("🔮 Analyze Activity"):
        with st.spinner("Analyzing..."):
            prediction = st.session_state.agent.predict_co2(f"{activity}|{category}")

            try:
                emission_value = float(prediction.split(": ")[1].split(" kg")[0])
            except:
                emission_value = 0

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Activity", activity)
            with col2:
                st.metric("Category", category)
            with col3:
                st.metric("CO2 Emission", f"{emission_value:.2f} kg", 
                         delta=None if emission_value == 0 else f"-{emission_value * 0.1:.2f} kg potential saving")

            emission_level = "High" if emission_value > 4 else "Medium" if emission_value > 1 else "Low"
            recommendations = st.session_state.agent.get_recommendations(f"{activity}|{category}|{emission_level}")

            st.markdown("---")
            st.markdown("### 💡 Personalized Recommendations")
            st.markdown(f'<div class="tip-card">{recommendations}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📈 Emission Comparison")

            comparison_data = {
                "Activity": ["Car(20km)", "Bus(20km)", "Bicycle(20km)", activity],
                "Emission": [4.6, 1.2, 0.0, emission_value],
                "Type": ["Reference", "Reference", "Reference", "Your Activity"]
            }
            df_comp = pd.DataFrame(comparison_data)

            fig = px.bar(df_comp, x="Activity", y="Emission", color="Type",
                        title="Your Activity vs. Common Activities",
                        color_discrete_map={"Reference": "#90CAF9", "Your Activity": "#FF6B6B"})
            st.plotly_chart(fig, use_container_width=True)

def show_dataset_analysis():
    st.markdown('<h2 class="sub-header">📊 Dataset Analysis</h2>', unsafe_allow_html=True)

    if not st.session_state.agent:
        st.warning("⚠️ Agent not initialized. Please check the model files.")
        return

    st.write("Upload your activity dataset (CSV format) for comprehensive analysis.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        temp_path = "temp_upload.csv"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        df = pd.read_csv(temp_path)

        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### 📈 Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total CO2", f"{df['AVG CO2 emission'].sum():.2f} kg")
        with col3:
            st.metric("Average CO2", f"{df['AVG CO2 emission'].mean():.2f} kg")
        with col4:
            st.metric("Max CO2", f"{df['AVG CO2 emission'].max():.2f} kg")

        st.markdown("---")
        st.markdown("### 📊 Visualizations")
        plot_emissions(df)

        if st.button("🤖 Get AI Recommendations"):
            with st.spinner("AI Agent analyzing your data..."):
                response = st.session_state.agent.query("Analyze dataset", dataset_path=temp_path)

                st.markdown("---")
                st.markdown("### 🤖 AI Analysis Report")

                if response["analysis"]:
                    st.text(response["analysis"])

                if response["recommendations"]:
                    st.markdown("### 💡 Top Recommendations")
                    st.markdown(f'<div class="tip-card">{response["recommendations"]}</div>', unsafe_allow_html=True)

        if os.path.exists(temp_path):
            os.remove(temp_path)

def show_tips():
    st.markdown('<h2 class="sub-header">💡 Sustainability Tips Database</h2>', unsafe_allow_html=True)

    st.write("Browse our comprehensive collection of sustainability tips organized by category.")

    try:
        with open('data/sustainability_tips.json', 'r') as f:
            tips = json.load(f)

        categories = list(set([tip['category'] for tip in tips]))
        selected_category = st.selectbox("Filter by Category:", ["All"] + categories)

        filtered_tips = tips if selected_category == "All" else [tip for tip in tips if tip['category'] == selected_category]

        st.markdown(f"### Showing {len(filtered_tips)} tips")

        for idx, tip in enumerate(filtered_tips):
            with st.expander(f"💡 {tip['category']} - {tip['activity']} ({tip['emission_level']} Emission)"):
                st.markdown(f"**Tip:** {tip['tip']}")
                st.markdown(f"**Impact:** {tip['impact']}")
                st.markdown(f"**Category:** {tip['category']}")

    except FileNotFoundError:
        st.error("Tips database not found. Please run the training notebook first.")

def show_about():
    st.markdown('<h2 class="sub-header">ℹ️ About Sustainability Advisor</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 🌍 Our Mission

    Sustainability Advisor is an AI-powered platform designed to help individuals and organizations 
    reduce their carbon footprint through data-driven insights and personalized recommendations.

    ### 🛠️ Technology Stack

    - **Machine Learning:** Random Forest Regressor for CO2 prediction
    - **Agentic AI:** LangChain for intelligent agent workflows
    - **Vector Store:** ChromaDB for efficient similarity search
    - **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
    - **UI Framework:** Streamlit for interactive web interface
    - **Visualization:** Plotly for interactive charts

    ### 📊 Features

    1. **CO2 Emission Prediction:** Accurate predictions using trained ML models
    2. **Smart Recommendations:** Context-aware tips using vector embeddings
    3. **Dataset Analysis:** Bulk processing of activity data
    4. **Interactive Visualizations:** Charts and graphs for better insights
    5. **Agentic Workflow:** Intelligent routing of user queries

    ### 🎓 How It Works

    1. **Data Collection:** Upload your activity data or enter single activities
    2. **AI Processing:** Our agent analyzes the data using ML models
    3. **Vector Search:** Relevant tips are retrieved using semantic similarity
    4. **Recommendations:** Personalized suggestions based on your carbon footprint

    ### 📧 Contact

    For questions, suggestions, or contributions, please reach out!

    ---

    Made with 💚 for a sustainable future 🌱
    """)

if __name__ == "__main__":
    main()
