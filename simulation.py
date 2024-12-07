# simulation.py
import numpy as np
from scipy import stats
from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MatchSimulation:
    """Simulate match outcomes using Monte Carlo methods"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.n_simulations = 1000
        
    def simulate_match(self, home_team_id: int, away_team_id: int,
                      league_id: int, season: int) -> Dict:
        """Run multiple simulations for a single match"""
        simulations = []
        
        for _ in range(self.n_simulations):
            sim_result = self._run_single_simulation(
                home_team_id, away_team_id, league_id, season
            )
            simulations.append(sim_result)
        
        return self._analyze_simulations(simulations)
    
    def _run_single_simulation(self, home_team_id: int, away_team_id: int,
                             league_id: int, season: int) -> Dict:
        """Run a single match simulation"""
        # Get base prediction
        base_pred = self.predictor.predict_match(
            home_team_id, away_team_id, league_id, season
        )
        
        # Add random variance
        home_goals = self._simulate_goals(base_pred['home_xg'])
        away_goals = self._simulate_goals(base_pred['away_xg'])
        
        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'result': 'H' if home_goals > away_goals else 'A' if away_goals > home_goals else 'D'
        }
    
    def _simulate_goals(self, xg: float) -> int:
        """Simulate number of goals using Poisson distribution"""
        return np.random.poisson(xg)
    
    def _analyze_simulations(self, simulations: List[Dict]) -> Dict:
        """Analyze simulation results"""
        results_count = {'H': 0, 'D': 0, 'A': 0}
        score_count = {}
        
        for sim in simulations:
            # Count results
            results_count[sim['result']] += 1
            
            # Count scores
            score = f"{sim['home_goals']}-{sim['away_goals']}"
            score_count[score] = score_count.get(score, 0) + 1
        
        return {
            'probability_distribution': {
                k: v/self.n_simulations for k, v in results_count.items()
            },
            'most_likely_scores': dict(sorted(
                score_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'confidence_intervals': self._calculate_confidence_intervals(simulations)
        }
    
    def _calculate_confidence_intervals(self, simulations: List[Dict]) -> Dict:
        """Calculate 95% confidence intervals for goals"""
        home_goals = [s['home_goals'] for s in simulations]
        away_goals = [s['away_goals'] for s in simulations]
        
        return {
            'home_goals': {
                'mean': np.mean(home_goals),
                'ci': stats.norm.interval(
                    0.95,
                    loc=np.mean(home_goals),
                    scale=stats.sem(home_goals)
                )
            },
            'away_goals': {
                'mean': np.mean(away_goals),
                'ci': stats.norm.interval(
                    0.95,
                    loc=np.mean(away_goals),
                    scale=stats.sem(away_goals)
                )
            }
        }

# Add daily matches prediction functionality
class DailyPredictions:
    """Handle daily match predictions for specified leagues"""
    
    def __init__(self, predictor, leagues=[39, 140, 135, 345]):
        self.predictor = predictor
        self.leagues = leagues
    
    def get_daily_matches(self) -> List[Dict]:
        """Fetch matches scheduled for today"""
        matches = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        for league_id in self.leagues:
            league_matches = self.predictor.data_collector.get_fixtures_by_date(
                league_id, today
            )
            if league_matches:
                matches.extend(league_matches)
        
        return matches
    
    def predict_daily_matches(self) -> List[Dict]:
        """Generate predictions for all matches today"""
        matches = self.get_daily_matches()
        predictions = []
        
        for match in matches:
            try:
                prediction = self.predictor.predict_match(
                    match['home_team']['id'],
                    match['away_team']['id'],
                    match['league_id'],
                    match['season']
                )
                
                predictions.append({
                    'match_id': match['fixture_id'],
                    'home_team': match['home_team']['name'],
                    'away_team': match['away_team']['name'],
                    'league_id': match['league_id'],
                    'kickoff': match['kickoff'],
                    'prediction': prediction['prediction'],
                    'probabilities': prediction['probabilities'],
                    'confidence': prediction['confidence']
                })
                
            except Exception as e:
                logger.error(f"Error predicting match {match['fixture_id']}: {str(e)}")
                continue
        
        # Sort by confidence
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    def display_predictions(self, predictions: List[Dict]):
        """Display predictions in a formatted way"""
        print("\nToday's Match Predictions")
        print("========================")
        
        confidence_levels = {
            'High Confidence (>75%)': [],
            'Medium Confidence (50-75%)': [],
            'Low Confidence (<50%)': []
        }
        
        for pred in predictions:
            if pred['confidence'] > 0.75:
                confidence_levels['High Confidence (>75%)'].append(pred)
            elif pred['confidence'] > 0.5:
                confidence_levels['Medium Confidence (50-75%)'].append(pred)
            else:
                confidence_levels['Low Confidence (<50%)'].append(pred)
        
        for level, preds in confidence_levels.items():
            if preds:
                print(f"\n{level}")
                print("-" * len(level))
                for pred in preds:
                    print(f"\n{pred['home_team']} vs {pred['away_team']}")
                    print(f"Prediction: {pred['prediction']}")
                    print(f"Confidence: {pred['confidence']:.1%}")
                    print(f"Kickoff: {pred['kickoff']}")