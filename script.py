import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
import pickle
import logging
import os
from dotenv import load_dotenv

from historical_analyzer import HistoricalAnalyzer
from nlp_analyzer import NLPAnalyzer

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

class AIPipeline:
    def __init__(self):
        self.historical_analyzer = HistoricalAnalyzer()
        self.nlp_analyzer = NLPAnalyzer()
        
    def analyze_match(self, home_team_id: int, away_team_id: int, 
                     home_stats: Dict, away_stats: Dict, h2h_matches: List[Dict]) -> Dict:
        """Analyze a match using all available components"""
        
        # Historical Analysis
        h2h_analysis = self.historical_analyzer.analyze_head_to_head(h2h_matches)
        home_form = self.historical_analyzer.analyze_team_form(home_stats)
        away_form = self.historical_analyzer.analyze_team_form(away_stats)
        
        # NLP Analysis
        home_news = self.nlp_analyzer.analyze_team_news(home_team_id)
        away_news = self.nlp_analyzer.analyze_team_news(away_team_id)
        home_lineup = self.nlp_analyzer.analyze_injuries_and_suspensions(home_team_id)
        away_lineup = self.nlp_analyzer.analyze_injuries_and_suspensions(away_team_id)
        
        # Calculate base probabilities from historical data
        base_probs = self._calculate_base_probabilities(
            home_form, away_form, h2h_analysis
        )
        
        # Adjust probabilities based on news and lineup analysis
        adjusted_probs = self._adjust_probabilities(
            base_probs,
            home_news, away_news,
            home_lineup, away_lineup
        )
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            h2h_analysis, home_form, away_form,
            home_news, away_news,
            home_lineup, away_lineup
        )
        
        return {
            'probabilities': adjusted_probs,
            'confidence': confidence,
            'analysis': {
                'h2h': h2h_analysis,
                'home_form': home_form,
                'away_form': away_form,
                'news_sentiment': {
                    'home': home_news,
                    'away': away_news
                },
                'lineup_impact': {
                    'home': home_lineup,
                    'away': away_lineup
                }
            }
        }

class DataCollector:
    """Handle all API interactions and data collection"""
    
    def __init__(self):
        # API credentials
        self.RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
        self.API_FOOTBALL_BASE = os.getenv('API_FOOTBALL_BASE')
        
        if not self.RAPIDAPI_KEY:
            raise ValueError("RAPIDAPI_KEY not found in environment variables")
            
        if not self.API_FOOTBALL_BASE:
            self.API_FOOTBALL_BASE = "https://api-football-v1.p.rapidapi.com/v3"
        
        # Headers for API requests
        self.headers = {
            "x-rapidapi-key": self.RAPIDAPI_KEY,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }
        self.request_delay = 1.0  # 1 second delay between requests
        self.last_request_time = 0
        
    def get_live_matches(self) -> List[Dict]:
        """Fetch today's matches"""
        leagues = [39, 140, 135, 345]  # Premier League, La Liga, Serie A, Czech League
        today = datetime.now().strftime('%Y-%m-%d')
        all_matches = []
        
        for league_id in leagues:
            matches = self.get_fixtures_by_date(league_id, today)
            if matches:
                all_matches.extend(matches)
        
        return all_matches

    def test_connection(self):
        """Test API connection and print response"""
        leagues = [39, 140, 135, 345]  # Premier League, La Liga, Serie A, Czech League
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate current season
        today_obj = datetime.now()
        season = today_obj.year
        if today_obj.month < 7:
            season -= 1
        
        print(f"Using season: {season}")
        
        for league_id in leagues:
            print(f"\nTesting league {league_id} for date {today}")
            endpoint = f"{self.API_FOOTBALL_BASE}/fixtures"
            params = {
                "league": league_id,
                "date": today,
                "season": season
            }
            
            try:
                response = requests.get(endpoint, headers=self.headers, params=params)
                print(f"Status Code: {response.status_code}")
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
            except Exception as e:
                print(f"Error: {str(e)}")
            
    def get_team_statistics(self, team_id: int, league_id: int, season: int) -> Dict:
        """Fetch team statistics from API-Football"""
        time.sleep(1)  
        endpoint = f"{self.API_FOOTBALL_BASE}/teams/statistics"
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            print(f"Raw team stats response: {json.dumps(data, indent=2)}")  # Debug line
            return data.get('response', {})
        except Exception as e:
            logger.error(f"Error fetching team statistics: {str(e)}")
            return None

    def get_fixtures_by_date(self, league_id: int, date: str) -> List[Dict]:
        """Fetch fixtures for a specific date"""
        endpoint = f"{self.API_FOOTBALL_BASE}/fixtures"
        
        # Calculate current season based on date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        season = date_obj.year
        # If before July, use previous year as season
        if date_obj.month < 7:
            season -= 1
        
        params = {
            "league": league_id,
            "date": date,
            "season": season
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data['response']
        except Exception as e:
            logger.error(f"Error fetching fixtures: {str(e)}")
            return []

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

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make a rate-limited API request"""
        # Ensure at least request_delay seconds since last request
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            self.last_request_time = time.time()
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {str(e)}")
            return None
        
        
class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def create_team_features(self, team_stats: Dict) -> pd.DataFrame:
        """Create advanced team features from raw statistics"""
        if not team_stats or 'fixtures' not in team_stats:
            return pd.DataFrame()
            
        try:
            fixtures = team_stats['fixtures']
            goals = team_stats['goals']
            
            features = {
                # Basic stats
                'games_played': fixtures['played']['total'],
                'wins': fixtures['wins']['total'],
                'draws': fixtures['draws']['total'],
                'losses': fixtures['loses']['total'],
                
                # Goals
                'goals_scored': goals['for']['total']['total'],
                'goals_conceded': goals['against']['total']['total'],
                
                # Clean sheets
                'clean_sheets': team_stats['clean_sheet']['total'],
                
                # Form calculation from last 5 matches
                'recent_form': sum(1 for x in team_stats['form'][:5] if x == 'W') / 5.0
            }
            
            # Calculate derived statistics
            if features['games_played'] > 0:
                features.update({
                    'win_rate': features['wins'] / features['games_played'],
                    'goals_per_game': features['goals_scored'] / features['games_played'],
                    'conceded_per_game': features['goals_conceded'] / features['games_played'],
                    'clean_sheet_rate': features['clean_sheets'] / features['games_played']
                })
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error processing team stats: {str(e)}")
            return pd.DataFrame()

class PredictionModel:
    """Handle model training, evaluation, and prediction"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(objective='multi:softprob', random_state=42)
        }
        self.best_model = None
        self.feature_importance = None

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train multiple models and select the best performing one"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred)
            }
            
            logger.info(f"{name} model accuracy: {accuracy:.4f}")

        # Select best model
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        self.best_model = self.models[best_model_name]
        
        if best_model_name in ['rf', 'xgb']:
            self.feature_importance = pd.DataFrame(
                self.best_model.feature_importances_,
                columns=['importance']
            )
        
        return results

    def predict_match(self, home_team_id: int, away_team_id: int,
                 league_id: int, season: int) -> Dict:
        """Predict the result of a specific match"""
        try:
            # Load or create training data
            training_data = self._get_training_data(league_id, season)
            
            # Fit the scaler and model if not already fitted
            if not hasattr(self.feature_engineering.scaler, 'mean_'):
                X = training_data.drop('result', axis=1)
                y = training_data['result']
                self.feature_engineering.scaler.fit(X)
                self.prediction_model.train_and_evaluate(X, y, X, y)  # Simple train/test split for now
            
            # Make prediction
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
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'prediction': 'Unknown',
                'probabilities': {'home_win': 0.33, 'draw': 0.34, 'away_win': 0.33},
                'confidence': 0.34,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class FootballPredictor:
    """Main class to orchestrate the prediction system"""
    
    def __init__(self, debug=True):
        self.data_collector = DataCollector()
        self.feature_engineering = FeatureEngineering()
        self.prediction_model = PredictionModel()
        self.debug = debug
        
    def prepare_match_data(self, home_team_id: int, away_team_id: int,
                      league_id: int, season: int) -> pd.DataFrame:
        """Prepare all necessary data for match prediction"""
        # Collect data
        home_stats = self.data_collector.get_team_statistics(home_team_id, league_id, season)
        away_stats = self.data_collector.get_team_statistics(away_team_id, league_id, season)
        
        if not home_stats or not away_stats:
            logger.error("Could not fetch team statistics")
            return pd.DataFrame()
        
        # Create features
        home_features = self.feature_engineering.create_team_features(home_stats)
        away_features = self.feature_engineering.create_team_features(away_stats)
        
        if home_features.empty or away_features.empty:
            logger.error("Could not create team features")
            return pd.DataFrame()
        
        # Combine features
        match_features = pd.concat([
            home_features.add_prefix('home_'),
            away_features.add_prefix('away_')
        ], axis=1)
        
        return match_features

    def predict_match(self, home_team_id: int, away_team_id: int,
                 league_id: int, season: int) -> Dict:
        """Predict the result of a specific match"""
        try:
            match_features = self.prepare_match_data(
                home_team_id, away_team_id, league_id, season
            )
            
            if match_features.empty:
                return {
                    'prediction': 'Unknown',
                    'probabilities': {'home_win': 0.34, 'draw': 0.33, 'away_win': 0.33},
                    'confidence': 0.34,
                    'timestamp': datetime.now().isoformat()
                }
            
            # For first-time predictions, initialize with basic model
            if not hasattr(self.prediction_model, 'best_model') or self.prediction_model.best_model is None:
                # Train a simple model with current data
                y = pd.Series(['H'])  # Dummy target for initialization
                self.prediction_model.train_and_evaluate(match_features, y, match_features, y)
            
            scaled_features = self.feature_engineering.scaler.fit_transform(match_features)
            prediction, probabilities = self.prediction_model.predict_match(scaled_features)
            
            return {
                'prediction': prediction,
                'probabilities': probabilities,
                'confidence': max(probabilities.values()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'prediction': 'Unknown',
                'probabilities': {'home_win': 0.34, 'draw': 0.33, 'away_win': 0.33},
                'confidence': 0.34,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    # Example: Premier League Match Prediction
    PREMIER_LEAGUE_ID = 39  # Premier League ID
    CURRENT_SEASON = 2024
    
    # Example team IDs
    ARSENAL_ID = 42
    CHELSEA_ID = 49
    
    predictor = FootballPredictor()
    
    try:
        prediction = predictor.predict_match(
            home_team_id=ARSENAL_ID,
            away_team_id=CHELSEA_ID,
            league_id=PREMIER_LEAGUE_ID,
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