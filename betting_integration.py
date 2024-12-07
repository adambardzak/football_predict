# betting_integration.py
import pandas as pd
import numpy as np
from typing import Dict, List
import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BettingIntegration:
    """Handle betting odds collection and analysis"""
    
    def __init__(self):
        self.ODDS_API_KEY = os.getenv('ODDS_API_KEY')
        self.BETFAIR_API_KEY = os.getenv('BETFAIR_API_KEY')
        
        # European bookmakers to track
        self.BOOKMAKERS = [
            'Bet365',
            'Unibet',
            'William Hill',
            'Bwin',
            'Pinnacle'
        ]
        
        # League-specific margin adjustments
        self.LEAGUE_MARGINS = {
            39: 1.05,  # Premier League
            140: 1.06,  # La Liga
            135: 1.06,  # Serie A
            345: 1.08   # Czech First League
        }

    def get_match_odds(self, match_id: int, league_id: int) -> Dict:
        """Fetch and analyze betting odds"""
        odds_data = self._fetch_odds(match_id)
        if not odds_data:
            return None
            
        processed_odds = self._process_odds(odds_data, league_id)
        value_bets = self._identify_value_bets(processed_odds)
        
        return {
            'market_odds': processed_odds,
            'value_bets': value_bets,
            'liquidity_score': self._calculate_liquidity_score(odds_data),
            'market_confidence': self._calculate_market_confidence(odds_data)
        }
    
    def _calculate_true_probabilities(self, odds: Dict, league_id: int) -> Dict:
        """Calculate true probabilities accounting for league-specific margins"""
        margin = self.LEAGUE_MARGINS.get(league_id, 1.07)  # Default margin
        implied_probs = {
            outcome: 1/price for outcome, price in odds.items()
        }
        total_prob = sum(implied_probs.values())
        
        # Adjust for margin
        true_probs = {
            outcome: prob/total_prob/margin 
            for outcome, prob in implied_probs.items()
        }
        
        return true_probs

    def _identify_value_bets(self, processed_odds: Dict) -> List[Dict]:
        """Identify valuable betting opportunities"""
        value_bets = []
        model_probs = processed_odds['model_probabilities']
        market_odds = processed_odds['best_odds']
        
        for outcome, prob in model_probs.items():
            if outcome in market_odds:
                ev = (prob * market_odds[outcome]) - 1
                if ev > 0.05:  # 5% edge threshold
                    value_bets.append({
                        'outcome': outcome,
                        'odds': market_odds[outcome],
                        'model_prob': prob,
                        'expected_value': ev,
                        'kelly_stake': self._kelly_criterion(prob, market_odds[outcome])
                    })
        
        return value_bets
    
    def _kelly_criterion(self, prob: float, odds: float, fraction: float = 0.5) -> float:
        """Calculate Kelly stake with fractional sizing"""
        b = odds - 1
        q = 1 - prob
        kelly = (b * prob - q) / b
        return max(0, kelly * fraction)  # Using half Kelly for conservative sizing

    def get_historical_odds_movement(self, match_id: int) -> Dict:
        """Track odds movement over time"""
        endpoint = f"{self.API_BASE}/odds/history/{match_id}"
        odds_history = self._make_api_request(endpoint)
        
        if not odds_history:
            return None
            
        return self._analyze_odds_movement(odds_history)
    
    def _analyze_odds_movement(self, odds_history: List) -> Dict:
        """Analyze odds movement patterns"""
        movements = {
            'home_win': [],
            'draw': [],
            'away_win': []
        }
        
        for odds in odds_history:
            timestamp = odds['timestamp']
            for outcome in movements.keys():
                movements[outcome].append({
                    'timestamp': timestamp,
                    'odds': odds[outcome],
                    'volume': odds.get('volume', 0)
                })
        
        return {
            'movements': movements,
            'sharp_money': self._detect_sharp_money(movements),
            'market_sentiment': self._analyze_market_sentiment(movements)
        }
    
    def _detect_sharp_money(self, movements: Dict) -> List[Dict]:
        """Detect sharp money movements"""
        sharp_moves = []
        
        for outcome, moves in movements.items():
            for i in range(1, len(moves)):
                odds_change = moves[i]['odds'] - moves[i-1]['odds']
                volume = moves[i]['volume']
                
                # Significant odds movement with high volume
                if abs(odds_change) > 0.1 and volume > 10000:
                    sharp_moves.append({
                        'timestamp': moves[i]['timestamp'],
                        'outcome': outcome,
                        'odds_change': odds_change,
                        'volume': volume
                    })
        
        return sharp_moves