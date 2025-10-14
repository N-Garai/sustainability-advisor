"""
Agentic AI Module for Sustainability Recommendations
Uses LangChain to create an intelligent agent for CO2 prediction and recommendations
"""

import os
import json
import pandas as pd
import joblib
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class SustainabilityAgent:
    """Agent for CO2 prediction and sustainability recommendations"""

    def __init__(self, model_path, encoders_path, tips_json_path):
        """Initialize the agent with models and vector store"""

        self.model = joblib.load(model_path)
        self.label_encoder_activity = joblib.load(f"{encoders_path}/label_encoder_activity.pkl")
        self.label_encoder_category = joblib.load(f"{encoders_path}/label_encoder_category.pkl")

        with open(f"{encoders_path}/activity_category_mapping.json", 'r') as f:
            self.activity_mapping = json.load(f)

        with open(tips_json_path, 'r') as f:
            self.sustainability_tips = json.load(f)

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self._initialize_vector_store()
        self._initialize_agent()

    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store with sustainability tips"""
        try:
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))

            self.collection = self.chroma_client.create_collection(
                name='sustainability_tips',
                metadata={'description': 'Sustainability tips for CO2 reduction'}
            )

            for idx, tip_data in enumerate(self.sustainability_tips):
                text_to_embed = f"{tip_data['category']} {tip_data['activity']} {tip_data['emission_level']}: {tip_data['tip']}"
                embedding = self.embedding_model.encode(text_to_embed).tolist()

                self.collection.add(
                    embeddings=[embedding],
                    documents=[tip_data['tip']],
                    metadatas=[{
                        'category': tip_data['category'],
                        'activity': tip_data['activity'],
                        'emission_level': tip_data['emission_level'],
                        'impact': tip_data['impact']
                    }],
                    ids=[f'tip_{idx}']
                )

            print("✓ Vector store initialized with sustainability tips")

        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")

    def _initialize_agent(self):
        """Initialize LangChain agent with tools"""

        tools = [
            Tool(
                name="Predict_CO2_Emission",
                func=self.predict_co2,
                description="Predicts CO2 emission for a given activity and category. Input should be in format: 'activity|category' e.g., 'Car(20km)|Transport'"
            ),
            Tool(
                name="Get_Sustainability_Tips",
                func=self.get_recommendations,
                description="Retrieves relevant sustainability tips based on activity, category, and emission level. Input should be in format: 'activity|category|emission_level'"
            ),
            Tool(
                name="Process_Dataset",
                func=self.process_dataset,
                description="Processes uploaded dataset and provides aggregated CO2 statistics and recommendations. Input should be a CSV file path."
            )
        ]

        template = """You are a sustainability expert AI agent helping users reduce their carbon footprint.

You have access to the following tools:

{tools}

Tool Names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question with detailed sustainability recommendations

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )

        self.tools = tools
        self.prompt = prompt

    def predict_co2(self, input_str):
        """Predict CO2 emission for given activity"""
        try:
            activity, category = input_str.split('|')

            activity_encoded = self.label_encoder_activity.transform([activity])[0]
            category_encoded = self.label_encoder_category.transform([category])[0]

            prediction = self.model.predict([[activity_encoded, category_encoded]])[0]

            return f"Predicted CO2 emission for {activity} in {category}: {prediction:.2f} kg"

        except Exception as e:
            return f"Error in prediction: {str(e)}"

    def get_recommendations(self, input_str):
        """Get sustainability recommendations from vector store"""
        try:
            parts = input_str.split('|')
            if len(parts) == 3:
                activity, category, emission_level = parts
            else:
                activity, category = parts
                emission_level = "All"

            query_text = f"{category} {activity} {emission_level}"
            query_embedding = self.embedding_model.encode(query_text).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            recommendations = []
            for i in range(len(results['documents'][0])):
                tip = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                recommendations.append(
                    f"💡 {tip}\n   📊 Impact: {metadata['impact']}"
                )

            return "\n\n".join(recommendations)

        except Exception as e:
            return f"Error getting recommendations: {str(e)}"

    def process_dataset(self, file_path):
        """Process uploaded dataset and provide analysis"""
        try:
            df = pd.read_csv(file_path)

            total_emission = df['AVG CO2 emission'].sum()
            avg_emission = df['AVG CO2 emission'].mean()

            category_stats = df.groupby('Category')['AVG CO2 emission'].agg(['sum', 'mean']).to_dict()
            activity_stats = df.groupby('Activity')['AVG CO2 emission'].agg(['sum', 'mean']).to_dict()

            result = f"""
Dataset Analysis:
================
Total CO2 Emission: {total_emission:.2f} kg
Average CO2 Emission: {avg_emission:.2f} kg

Category Breakdown:
{json.dumps(category_stats, indent=2)}

Top High-Emission Activities:
{df.nlargest(5, 'AVG CO2 emission')[['Activity', 'AVG CO2 emission']].to_string()}
"""
            return result

        except Exception as e:
            return f"Error processing dataset: {str(e)}"

    def query(self, user_input, dataset_path=None):
        """Process user query and return response"""

        response = {"prediction": None, "recommendations": None, "analysis": None}

        if dataset_path:
            response["analysis"] = self.process_dataset(dataset_path)

        if "predict" in user_input.lower() or "emission" in user_input.lower():
            if "|" in user_input:
                parts = user_input.split("|")
                if len(parts) >= 2:
                    response["prediction"] = self.predict_co2(f"{parts[0]}|{parts[1]}")

        if dataset_path:
            df = pd.read_csv(dataset_path)
            most_common = df['Activity'].mode()[0]
            category = df[df['Activity'] == most_common]['Category'].iloc[0]
            emission = df[df['Activity'] == most_common]['AVG CO2 emission'].iloc[0]

            emission_level = "High" if emission > 4 else "Medium" if emission > 1 else "Low"
            response["recommendations"] = self.get_recommendations(f"{most_common}|{category}|{emission_level}")
        else:
            response["recommendations"] = self.get_recommendations("All|General|All")

        return response

def create_agent(model_path, encoders_path, tips_json_path):
    """Factory function to create sustainability agent"""
    return SustainabilityAgent(model_path, encoders_path, tips_json_path)
