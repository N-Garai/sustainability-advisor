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


# Page configuration
st.set_page_config(
    page_title="🌍 Sustainability Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED CSS - Proper colors for both light and dark modes
# UPDATED CSS - Replace the entire st.markdown CSS section with this:
st.markdown("""
<style>
    /* Main headers - dark green, bold */
    .main-header {
        font-size: 2.5rem;
        color: #1B5E20 !important;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    
    /* Sub headers - medium green */
    .sub-header {
        font-size: 1.8rem;
        color: #2E7D32 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Tip cards - light green background with dark text */
    .tip-card {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        color: #1B5E20 !important;
    }
    
    .tip-card h3 {
        color: #2E7D32 !important;
        margin-bottom: 0.5rem;
    }
    
    .tip-card p {
        color: #33691E !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        color: #E65100 !important;
        border: 2px solid #FFB74D;
    }
    
    .metric-card h3 {
        color: #E65100 !important;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #F57C00 !important;
        font-size: 0.95rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        background-color: #45a049 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Ensure all text is visible */
    div[data-testid="stMarkdownContainer"] p {
        color: #212121 !important;
    }
    
    /* ===== SIDEBAR STYLING - FIXED ===== */
    section[data-testid="stSidebar"] {
        background-color: #E8F5E9;
    }
    
    /* Sidebar text - dark green */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #1B5E20 !important;
    }
    
    /* Sidebar title */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #2E7D32 !important;
    }
    
    /* Sidebar radio buttons text */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #1B5E20 !important;
        font-weight: 500;
    }
    
    /* Sidebar metrics */
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {
        color: #2E7D32 !important;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: #1B5E20 !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    section[data-testid="stSidebar"] div[data-testid="stMetricDelta"] {
        color: #4CAF50 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #E8F5E9 !important;
        color: #1B5E20 !important;
        font-weight: 600;
    }
    
    /* Table styling */
    .dataframe {
        color: #212121 !important;
    }
    
    /* Main content metrics */
    div[data-testid="stMetricValue"] {
        color: #1B5E20 !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #424242 !important;
        font-weight: 500;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #81C784 !important;
        }
        
        .sub-header {
            color: #66BB6A !important;
        }
        
        div[data-testid="stMarkdownContainer"] p {
            color: #E0E0E0 !important;
        }
        
        .tip-card {
            background-color: #1B5E20;
            color: #E8F5E9 !important;
        }
        
        .tip-card h3 {
            color: #A5D6A7 !important;
        }
        
        .tip-card p {
            color: #C8E6C9 !important;
        }
        
        .metric-card {
            background-color: #4E342E;
            color: #FFCCBC !important;
        }
        
        .metric-card h3 {
            color: #FFB74D !important;
        }
        
        .metric-card p {
            color: #FFCCBC !important;
        }
        
        section[data-testid="stSidebar"] {
            background-color: #1B5E20;
        }
        
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #C8E6C9 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def initialize_agent():
    """Initialize the sustainability agent"""
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
    """Create visualizations for emissions data with HIGH CONTRAST"""
    
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
        
        # FIXED: Dark, bold text for ALL chart elements
        fig1.update_layout(
            title={
                'text': '<b>Average CO2 Emissions by Activity</b>',
                'font': {'size': 20, 'color': '#1B5E20', 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color='#212121'  # Dark text
            ),
            paper_bgcolor='white',
            plot_bgcolor='#FAFAFA',
            xaxis=dict(
                title={
                    'text': '<b>Activity</b>',
                    'font': {'size': 16, 'color': '#212121'}
                },
                tickfont={'size': 13, 'color': '#212121'},
                gridcolor='#E0E0E0',
                showgrid=True
            ),
            yaxis=dict(
                title={
                    'text': '<b>Emission (kg CO2)</b>',
                    'font': {'size': 16, 'color': '#212121'}
                },
                tickfont={'size': 13, 'color': '#212121'},
                gridcolor='#E0E0E0',
                showgrid=True
            ),
            coloraxis_colorbar=dict(
                title={
                    'text': '<b>CO2 (kg)</b>',
                    'font': {'color': '#212121'}
                },
                tickfont={'color': '#212121'},
                bgcolor='white'
            ),
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        category_sum = df.groupby('Category')['AVG CO2 emission'].sum()
        
        fig2 = px.pie(
            values=category_sum.values,
            names=category_sum.index,
            title='CO2 Emissions Distribution by Category',
            color_discrete_sequence=['#66BB6A', '#FFA726']  # Green and Orange for contrast
        )
        
        # FIXED: Dark, bold text for pie chart
        fig2.update_layout(
            title={
                'text': '<b>CO2 Emissions Distribution by Category</b>',
                'font': {'size': 20, 'color': '#1B5E20', 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color='#212121'
            ),
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                font={'size': 13, 'color': '#212121'},
                bgcolor='white',
                bordercolor='#E0E0E0',
                borderwidth=1
            ),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        # Make text on pie slices white and bold
        fig2.update_traces(
            textfont={'size': 15, 'color': 'white', 'family': 'Arial, sans-serif'},
            textposition='inside',
            texttemplate='%{label}<br>%{value:.1f} kg<br>(%{percent})',
            marker=dict(
                line=dict(color='white', width=3)
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)


def main():
    """Main application"""
    
    # Header with proper styling
    st.markdown('<h1 class="main-header">🌍 Sustainability Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #424242; font-size: 1.2rem; margin-bottom: 2rem;">Your Personal AI Agent for Carbon Footprint Reduction</p>', unsafe_allow_html=True)
    
    # Initialize agent
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing AI Agent..."):
            st.session_state.agent = initialize_agent()
            if st.session_state.agent:
                st.session_state.initialized = True
                st.success("✅ Agent initialized successfully!")
    
        # Sidebar with FIXED visibility
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/tree-planting.png", width=100)
    
    # Title with dark text
        st.markdown('<h1 style="color: #1B5E20;">Navigation</h1>', unsafe_allow_html=True)
    
    page = st.radio(
        "Choose a feature:",
        ["🏠 Home", "🔍 Single Activity Analysis", "📊 Dataset Analysis", "💡 Sustainability Tips", "ℹ️ About"]
    )
    
    st.markdown("---")
    
    # FIXED: Quick Stats with visible text
    st.markdown('<h3 style="color: #1B5E20;">Quick Stats</h3>', unsafe_allow_html=True)
    
    # Create custom metric cards with high contrast
    st.markdown("""
    <div style="background-color: #FFFFFF; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 2px solid #4CAF50;">
        <p style="color: #616161; font-size: 0.9rem; margin: 0;">CO2 Saved Today</p>
        <p style="color: #1B5E20; font-size: 1.8rem; font-weight: bold; margin: 0.2rem 0;">24.5 kg</p>
        <p style="color: #4CAF50; font-size: 0.9rem; margin: 0;">↑ +5.2 kg</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #FFFFFF; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 2px solid #4CAF50;">
        <p style="color: #616161; font-size: 0.9rem; margin: 0;">Trees Equivalent</p>
        <p style="color: #1B5E20; font-size: 1.8rem; font-weight: bold; margin: 0.2rem 0;">1.1</p>
        <p style="color: #4CAF50; font-size: 0.9rem; margin: 0;">↑ +0.2</p>
    </div>
    """, unsafe_allow_html=True)

    
    
    # Main content based on page selection
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
    """Home page"""
    st.markdown('<h2 class="sub-header">Welcome to Your Sustainability Journey! 🌱</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="metric-card">
            <h3>🎯 Track</h3>
            <p>Monitor your daily CO2 emissions</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="metric-card">
            <h3>📉 Reduce</h3>
            <p>Get personalized recommendations</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div class="metric-card">
            <h3>🌍 Impact</h3>
            <p>Make a real difference</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown('<h3 class="sub-header">🚀 Key Features</h3>', unsafe_allow_html=True)
    
    features = [
        {"icon": "🤖", "title": "AI-Powered Analysis", "desc": "Advanced machine learning models predict your carbon footprint"},
        {"icon": "📊", "title": "Dataset Processing", "desc": "Upload your activity data for comprehensive analysis"},
        {"icon": "💡", "title": "Smart Recommendations", "desc": "Personalized tips using vector search and embeddings"},
        {"icon": "📈", "title": "Visual Insights", "desc": "Interactive charts and graphs for better understanding"}
    ]
    
    cols = st.columns(2)
    for idx, feature in enumerate(features):
        with cols[idx % 2]:
            st.markdown(f'''
            <div class="tip-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            ''', unsafe_allow_html=True)


def show_single_analysis():
    """Single activity analysis page"""
    st.markdown('<h2 class="sub-header">🔍 Single Activity Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.agent:
        st.warning("⚠️ Agent not initialized. Please check the model files.")
        return
    
    st.markdown('<p style="color: #424242; font-size: 1.1rem;">Enter your activity details to get CO2 emission prediction and personalized recommendations.</p>', unsafe_allow_html=True)
    
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
            st.markdown('<h3 class="sub-header">📊 Prediction Results</h3>', unsafe_allow_html=True)
            
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
            st.markdown('<h3 class="sub-header">💡 Personalized Recommendations</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="tip-card" style="white-space: pre-line;">{recommendations}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📈 Emission Comparison</h3>', unsafe_allow_html=True)
            
            comparison_data = {
                "Activity": ["Car(20km)", "Bus(20km)", "Bicycle(20km)", activity],
                "Emission": [4.6, 1.2, 0.0, emission_value],
                "Type": ["Reference", "Reference", "Reference", "Your Activity"]
            }
            df_comp = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                df_comp, 
                x="Activity", 
                y="Emission", 
                color="Type",
                title="Your Activity vs. Common Activities",
                color_discrete_map={"Reference": "#81C784", "Your Activity": "#FF6B6B"}
            )
            
            # FIXED: High contrast labels for all chart elements
            fig.update_layout(
                title={
                    'text': '<b>Your Activity vs. Common Activities</b>',
                    'font': {'size': 20, 'color': '#1B5E20', 'family': 'Arial, sans-serif'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                font={
                    'family': 'Arial, sans-serif', 
                    'size': 14, 
                    'color': '#212121'
                },
                paper_bgcolor='white',
                plot_bgcolor='#FAFAFA',
                xaxis=dict(
                    title={
                        'text': '<b>Activity</b>', 
                        'font': {'size': 16, 'color': '#212121'}
                    },
                    tickfont={'size': 13, 'color': '#212121'},
                    gridcolor='#E0E0E0',
                    showgrid=True
                ),
                yaxis=dict(
                    title={
                        'text': '<b>Emission (kg CO2)</b>', 
                        'font': {'size': 16, 'color': '#212121'}
                    },
                    tickfont={'size': 13, 'color': '#212121'},
                    gridcolor='#E0E0E0',
                    showgrid=True
                ),
                legend=dict(
                    title={'text': '<b>Type</b>', 'font': {'size': 14, 'color': '#212121'}},
                    font={'size': 13, 'color': '#212121'},
                    bgcolor='white',
                    bordercolor='#E0E0E0',
                    borderwidth=1
                ),
                margin=dict(l=80, r=80, t=80, b=80)
            )
            
            st.plotly_chart(fig, use_container_width=True)


def show_dataset_analysis():
    """Dataset analysis page"""
    st.markdown('<h2 class="sub-header">📊 Dataset Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.agent:
        st.warning("⚠️ Agent not initialized. Please check the model files.")
        return
    
    st.markdown('<p style="color: #424242; font-size: 1.1rem;">Upload your activity dataset (CSV format) for comprehensive analysis.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        temp_path = "temp_upload.csv"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        df = pd.read_csv(temp_path)
        
        st.markdown('<h3 class="sub-header">📋 Dataset Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown('<h3 class="sub-header">📈 Statistical Summary</h3>', unsafe_allow_html=True)
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
        st.markdown('<h3 class="sub-header">📊 Visualizations</h3>', unsafe_allow_html=True)
        plot_emissions(df)
        
        if st.button("🤖 Get AI Recommendations"):
            with st.spinner("AI Agent analyzing your data..."):
                response = st.session_state.agent.query("Analyze dataset", dataset_path=temp_path)
                
                st.markdown("---")
                st.markdown('<h3 class="sub-header">🤖 AI Analysis Report</h3>', unsafe_allow_html=True)
                
                if response["analysis"]:
                    st.markdown(f'<pre style="color: #212121; background-color: #F5F5F5; padding: 1rem; border-radius: 5px;">{response["analysis"]}</pre>', unsafe_allow_html=True)
                
                if response["recommendations"]:
                    st.markdown('<h3 class="sub-header">💡 Top Recommendations</h3>', unsafe_allow_html=True)
                    st.markdown(f'<div class="tip-card" style="white-space: pre-line;">{response["recommendations"]}</div>', unsafe_allow_html=True)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)


def show_tips():
    """Sustainability tips page"""
    st.markdown('<h2 class="sub-header">💡 Sustainability Tips Database</h2>', unsafe_allow_html=True)
    
    st.markdown('<p style="color: #424242; font-size: 1.1rem;">Browse our comprehensive collection of sustainability tips organized by category.</p>', unsafe_allow_html=True)
    
    try:
        with open('data/sustainability_tips.json', 'r') as f:
            tips = json.load(f)
        
        categories = list(set([tip['category'] for tip in tips]))
        selected_category = st.selectbox("Filter by Category:", ["All"] + categories)
        
        filtered_tips = tips if selected_category == "All" else [tip for tip in tips if tip['category'] == selected_category]
        
        st.markdown(f'<h3 class="sub-header">Showing {len(filtered_tips)} tips</h3>', unsafe_allow_html=True)
        
        for idx, tip in enumerate(filtered_tips):
            with st.expander(f"💡 {tip['category']} - {tip['activity']} ({tip['emission_level']} Emission)"):
                st.markdown(f"**Tip:** {tip['tip']}")
                st.markdown(f"**Impact:** {tip['impact']}")
                st.markdown(f"**Category:** {tip['category']}")
        
    except FileNotFoundError:
        st.error("Tips database not found. Please run the training notebook first.")


def show_about():
    """About page"""
    st.markdown('<h2 class="sub-header">ℹ️ About Sustainability Advisor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: #212121; line-height: 1.8;">
    
    ### 🌍 Our Mission
    
    Sustainability Advisor is an AI-powered platform designed to help individuals and organizations 
    reduce their carbon footprint through data-driven insights and personalized recommendations.
    
    ### 🛠️ Technology Stack
    
    - **Machine Learning:** Random Forest Regressor for CO2 prediction
    - **Agentic AI:** Custom intelligent agent workflows
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
    
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
