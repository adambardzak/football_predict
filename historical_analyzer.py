import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class HistoricalAnalyzer:
    def __init__(self):
        self.max_history_days = 365 * 3  # 3 years of history
        self.recency_weight_factor = 365  # Weight halves every year
        
    def analyze_head_to_head(self, h2h_matches: List[Dict]) -> Dict:
        """Analyze head-to-head match history with recency weighting"""
        if not h2h_matches:
            return {
                'h2h_home_win_rate': 0.5,
                'h2h_away_win_rate': 0.5,
                'h2h_draw_rate': 0.0,
                'h2h_confidence': 0.0
            }
            
        weighted_results = {
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'total_weight': 0
        }
        
        for match in h2h_matches:
            weight = self._calculate_recency_weight(match['fixture']['date'])
            
            if match['goals']['home'] > match['goals']['away']:
                weighted_results['home_wins'] += weight
            elif match['goals']['home'] < match['goals']['away']:
                weighted_results['away_wins'] += weight
            else:
                weighted_results['draws'] += weight
                
            weighted_results['total_weight'] += weight
            
        total_weight = weighted_results['total_weight'] or 1
        
        return {
            'h2h_home_win_rate': weighted_results['home_wins'] / total_weight,
            'h2h_away_win_rate': weighted_results['away_wins'] / total_weight,
            'h2h_draw_rate': weighted_results['draws'] / total_weight,
            'h2h_confidence': min(total_weight / 5, 1.0)  # Confidence based on number of matches
        }
    
    def analyze_team_form(self, team_stats: Dict) -> Dict:
        """Analyze team's recent form and performance"""
        if not team_stats:
            return {
                'form_score': 0.5,
                'attack_strength': 0.5,
                'defense_strength': 0.5,
                'form_confidence': 0.0
            }
            
        # Calculate form from recent results
        form_string = team_stats.get('form', '')
        recent_form = form_string[:5]  # Last 5 matches
        
        form_score = sum(1 if result == 'W' else 0.5 if result == 'D' else 0 
                        for result in recent_form) / len(recent_form) if recent_form else 0.5
        
        # Calculate attack and defense strength
        fixtures = team_stats.get('fixtures', {})
        goals = team_stats.get('goals', {})
        
        games_played = fixtures.get('played', {}).get('total', 0)
        if games_played > 0:
            goals_scored = goals.get('for', {}).get('total', {}).get('total', 0)
            goals_conceded = goals.get('against', {}).get('total', {}).get('total', 0)
            
            attack_strength = goals_scored / games_played / 1.5  # Normalized to typical goal average
            defense_strength = 1 - (goals_conceded / games_played / 1.5)
        else:
            attack_strength = 0.5
            defense_strength = 0.5
        
        return {
            'form_score': form_score,
            'attack_strength': attack_strength,
            'defense_strength': defense_strength,
            'form_confidence': min(games_played / 10, 1.0)  # Confidence based on games played
        }
    
    def _calculate_recency_weight(self, match_date: str) -> float:
        """Calculate weight based on how recent the match was"""
        try:
            match_datetime = datetime.strptime(match_date, "%Y-%m-%d")
            days_ago = (datetime.now() - match_datetime).days
            
            if days_ago > self.max_history_days:
                return 0.0
                
            weight = np.exp(-days_ago / self.recency_weight_factor)
            return weight
            
        except Exception as e:
            logger.error(f"Error calculating recency weight: {str(e)}")
            return 0.0