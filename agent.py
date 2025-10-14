"""
Simplified Agent module without LangChain dependencies
All functionality preserved without external LangChain requirement
"""

import os
import json
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class SustainabilityAgent:
    """Agent for CO2 prediction and sustainability recommendations"""

    def __init__(self, model_path: str, encoders_path: str, tips_json_path: str):
        """Initialize agent with ML models and vector store"""
        
        # Load ML model and encoders
        self.model = joblib.load(model_path)
        self.label_encoder_activity = joblib.load(
            os.path.join(encoders_path, "label_encoder_activity.pkl")
        )
        self.label_encoder_category = joblib.load(
            os.path.join(encoders_path, "label_encoder_category.pkl")
        )

        # Load activity-category mapping
        mapping_path = os.path.join(encoders_path, "activity_category_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.activity_mapping = json.load(f)

        # Load sustainability tips
        with open(tips_json_path, "r", encoding="utf-8") as f:
            self.sustainability_tips = json.load(f)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize vector store
        self._initialize_vector_store()
        
        print("✓ Sustainability Agent initialized successfully")

    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store with sustainability tips"""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.Client(
                Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="sustainability_tips",
                metadata={"description": "Sustainability tips for CO2 reduction"}
            )

            # Check if collection needs to be populated
            try:
                count = self.collection.count()
                needs_population = (count == 0)
            except:
                needs_population = True

            # Populate vector store if empty
            if needs_population:
                self._populate_vector_store()
                
            print(f"✓ Vector store initialized with {self.collection.count()} tips")
            
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            raise

    def _populate_vector_store(self):
        """Populate vector store with sustainability tips"""
        docs, embeddings, metadatas, ids = [], [], [], []
        
        for idx, tip_data in enumerate(self.sustainability_tips):
            # Create embedding text
            text_to_embed = (
                f"{tip_data['category']} {tip_data['activity']} "
                f"{tip_data['emission_level']}: {tip_data['tip']}"
            )
            
            # Generate embedding
            embedding = self.embedding_model.encode(text_to_embed).tolist()
            
            # Prepare data
            docs.append(tip_data["tip"])
            embeddings.append(embedding)
            metadatas.append({
                "category": tip_data["category"],
                "activity": tip_data["activity"],
                "emission_level": tip_data["emission_level"],
                "impact": tip_data["impact"]
            })
            ids.append(f"tip_{idx}")
        
        # Add to collection
        if embeddings:
            self.collection.add(
                embeddings=embeddings,
                documents=docs,
                metadatas=metadatas,
                ids=ids
            )

    def predict_co2(self, input_str: str) -> str:
        """
        Predict CO2 emission for given activity.
        
        Args:
            input_str: Format 'Activity|Category' e.g., 'Car(20km)|Transport'
            
        Returns:
            Prediction string with CO2 emission value
        """
        try:
            activity, category = input_str.split("|", maxsplit=1)
            
            # Encode inputs
            activity_encoded = self.label_encoder_activity.transform([activity])[0]
            category_encoded = self.label_encoder_category.transform([category])[0]
            
            # Predict
            prediction = float(self.model.predict([[activity_encoded, category_encoded]])[0])
            
            return f"Predicted CO2 emission for {activity} in {category}: {prediction:.2f} kg"
            
        except Exception as e:
            return f"Error in prediction: {str(e)}"

    def get_recommendations(self, input_str: str) -> str:
        """
        Get sustainability recommendations from vector store.
        
        Args:
            input_str: Format 'Activity|Category|EmissionLevel' or 'Activity|Category'
            
        Returns:
            Formatted recommendations string
        """
        try:
            parts = input_str.split("|")
            
            if len(parts) == 3:
                activity, category, emission_level = parts
            else:
                activity, category = parts[0], parts[1]
                emission_level = "All"
            
            # Create query for vector search
            query_text = f"{category} {activity} {emission_level}"
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Search vector store
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            # Format recommendations
            recommendations = []
            for i in range(len(results["documents"][0])):
                tip = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                recommendations.append(
                    f"💡 {tip}\n   📊 Impact: {metadata.get('impact', 'N/A')}"
                )
            
            return "\n\n".join(recommendations) if recommendations else "No recommendations found."
            
        except Exception as e:
            return f"Error getting recommendations: {str(e)}"

    def process_dataset(self, file_path: str) -> str:
        """
        Process uploaded dataset and provide analysis.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Formatted analysis string
        """
        try:
            df = pd.read_csv(file_path)
            
            # Calculate statistics
            total_emission = float(df["AVG CO2 emission"].sum())
            avg_emission = float(df["AVG CO2 emission"].mean())
            
            # Breakdown by category
            category_stats = (
                df.groupby("Category")["AVG CO2 emission"]
                .agg(["sum", "mean"])
                .round(2)
                .to_dict()
            )
            
            # Top high-emission activities
            top_activities = df.nlargest(5, "AVG CO2 emission")[
                ["Activity", "AVG CO2 emission"]
            ]
            
            result = f"""
Dataset Analysis:
================
Total CO2 Emission: {total_emission:.2f} kg
Average CO2 Emission: {avg_emission:.2f} kg

Category Breakdown:
{json.dumps(category_stats, indent=2)}

Top High-Emission Activities:
{top_activities.to_string(index=False)}
"""
            return result
            
        except Exception as e:
            return f"Error processing dataset: {str(e)}"

    def query(self, user_input: str, dataset_path: str = None) -> dict:
        """
        Process user query and return comprehensive response.
        
        Args:
            user_input: User's query string
            dataset_path: Optional path to dataset CSV
            
        Returns:
            Dictionary with prediction, recommendations, and analysis
        """
        response = {
            "prediction": None,
            "recommendations": None,
            "analysis": None
        }
        
        # Analyze dataset if provided
        if dataset_path:
            response["analysis"] = self.process_dataset(dataset_path)
        
        # Handle prediction queries
        if "|" in user_input and any(
            keyword in user_input.lower() 
            for keyword in ["predict", "emission"]
        ):
            parts = user_input.split("|")
            if len(parts) >= 2:
                response["prediction"] = self.predict_co2(f"{parts[0]}|{parts[1]}")
        
        # Get recommendations
        if dataset_path:
            # Use most common activity from dataset
            df = pd.read_csv(dataset_path)
            most_common = df["Activity"].mode()[0]
            category = df[df["Activity"] == most_common]["Category"].iloc[0]
            emission = float(df[df["Activity"] == most_common]["AVG CO2 emission"].iloc[0])
            
            emission_level = "High" if emission > 4 else "Medium" if emission > 1 else "Low"
            response["recommendations"] = self.get_recommendations(
                f"{most_common}|{category}|{emission_level}"
            )
        else:
            # Default general recommendations
            response["recommendations"] = self.get_recommendations("All|General|All")
        
        return response


def create_agent(model_path: str, encoders_path: str, tips_json_path: str) -> SustainabilityAgent:
    """
    Factory function to create sustainability agent.
    
    Args:
        model_path: Path to trained model pickle file
        encoders_path: Path to directory containing encoder files
        tips_json_path: Path to sustainability tips JSON file
        
    Returns:
        Initialized SustainabilityAgent instance
    """
    return SustainabilityAgent(model_path, encoders_path, tips_json_path)
