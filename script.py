import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('football_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Handle all API interactions and data collection"""
    
    def __init__(self):
        # API credentials
        self.RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
        if not self.RAPIDAPI_KEY:
            raise ValueError("RAPIDAPI_KEY not found in environment variables")
        
        # API endpoints
        self.API_FOOTBALL_BASE = "https://api-football-v1.p.rapidapi.com/v3"
        
        # Headers for API requests
        self.headers = {
            "x-rapidapi-key": self.RAPIDAPI_KEY,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }

    def get_team_statistics(self, team_id: int, league_id: int, season: int) -> Dict:
        """Fetch team statistics from API-Football"""
        endpoint = f"{self.API_FOOTBALL_BASE}/teams/statistics"
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Error fetching team statistics: {str(e)}")
            return None

    def get_player_statistics(self, player_id: int, season: int) -> Dict:
        """Fetch player statistics from SportMonks"""
        endpoint = f"{self.SPORTMONKS_BASE}/players/{player_id}/statistics"
        params = {
            "api_token": self.SPORTMONKS_KEY,
            "season": season
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()['data']
        except Exception as e:
            logger.error(f"Error fetching player statistics: {str(e)}")
            return None

    def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10) -> List[Dict]:
        """Fetch head-to-head matches history"""
        endpoint = f"{self.API_FOOTBALL_BASE}/fixtures/headtohead"
        params = {
            "h2h": f"{team1_id}-{team2_id}",
            "last": limit
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Error fetching head-to-head data: {str(e)}")
            return None

class FootballPredictor:
    """Main class to orchestrate the prediction system"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineering = FeatureEngineering()
        self.prediction_model = PredictionModel()
        
    def prepare_match_data(self, home_team_id: int, away_team_id: int,
                          league_id: int, season: int) -> np.ndarray:
        """Prepare all necessary data for match prediction"""
        # Collect data
        home_stats = self.data_collector.get_team_statistics(home_team_id, league_id, season)
        away_stats = self.data_collector.get_team_statistics(away_team_id, league_id, season)
        h2h_history = self.data_collector.get_head_to_head(home_team_id, away_team_id)
        
        # Create features
        home_features = self.feature_engineering.create_team_features(home_stats)
        away_features = self.feature_engineering.create_team_features(away_stats)
        
        # Add head-to-head features
        h2h_features = self._create_h2h_features(h2h_history, home_team_id)
        
        # Combine all features
        match_features = pd.concat([
            home_features.add_prefix('home_'),
            away_features.add_prefix('away_'),
            h2h_features
        ], axis=1)
        
        return match_features.values

    def predict_match(self, home_team_id: int, away_team_id: int,
                     league_id: int, season: int) -> Dict:
        """Predict the result of a specific match"""
        match_features = self.prepare_match_data(
            home_team_id, away_team_id, league_id, season
        )
        
        scaled_features = self.feature_engineering.scaler.transform(match_features)
        prediction, probabilities = self.prediction_model.predict_match(scaled_features)
        
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': max(probabilities.values()),
            'timestamp': datetime.now().isoformat()
        }

def main():
    # Example: Czech League Match Prediction
    CZECH_LEAGUE_ID = 345  # This is an example ID, verify the correct one
    CURRENT_SEASON = 2024
    
    # Example Czech team IDs
    SLAVIA_PRAHA_ID = 558  # Example ID
    SPARTA_PRAHA_ID = 559  # Example ID
    
    predictor = FootballPredictor()
    
    try:
        prediction = predictor.predict_match(
            home_team_id=SLAVIA_PRAHA_ID,
            away_team_id=SPARTA_PRAHA_ID,
            league_id=CZECH_LEAGUE_ID,
            season=CURRENT_SEASON
        )
        
        print("\nMatch Prediction Report")
        print("=====================")
        print(f"Prediction: {prediction['prediction']}")
        print("\nProbabilities:")
        for outcome, prob in prediction['probabilities'].items():
            print(f"{outcome}: {prob:.2%}")
        print(f"\nConfidence: {prediction['confidence']:.2%}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()