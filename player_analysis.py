# player_analysis.py
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PlayerAnalysis:
    """Advanced player analysis for European football"""
    
    def __init__(self):
        self.LEAGUE_ADJUSTMENTS = {
            39: 1.0,    # Premier League (baseline)
            140: 1.1,   # La Liga (more technical)
            135: 1.05,  # Serie A (tactical)
            345: 1.15   # Czech First League (adjustment for competition level)
        }
        
    def analyze_squad_strength(self, team_id: int, league_id: int) -> Dict:
        """Analyze overall squad strength with league-specific adjustments"""
        player_stats = self._get_squad_statistics(team_id)
        if not player_stats:
            return None
            
        league_factor = self.LEAGUE_ADJUSTMENTS.get(league_id, 1.0)
        
        analysis = {
            'overall_strength': self._calculate_squad_strength(player_stats) * league_factor,
            'key_players': self._identify_key_players(player_stats),
            'fatigue_risk': self._assess_squad_fatigue(player_stats),
            'depth_analysis': self._analyze_squad_depth(player_stats),
            'form_analysis': self._analyze_squad_form(player_stats)
        }
        
        return analysis
    
    def _calculate_squad_strength(self, player_stats: List[Dict]) -> float:
        """Calculate overall squad strength based on player ratings"""
        weights = {
            'attack': 0.3,
            'midfield': 0.3,
            'defense': 0.25,
            'goalkeeper': 0.15
        }
        
        strengths = {
            'attack': self._calculate_attack_strength(player_stats),
            'midfield': self._calculate_midfield_strength(player_stats),
            'defense': self._calculate_defense_strength(player_stats),
            'goalkeeper': self._calculate_goalkeeper_strength(player_stats)
        }
        
        return sum(strength * weights[pos] for pos, strength in strengths.items())
    
    def _identify_key_players(self, player_stats: List[Dict]) -> List[Dict]:
        """Identify key players and their impact"""
        key_players = []
        
        for player in player_stats:
            impact_score = self._calculate_player_impact(player)
            if impact_score > 8.0:  # High impact threshold
                key_players.append({
                    'id': player['id'],
                    'name': player['name'],
                    'position': player['position'],
                    'impact_score': impact_score,
                    'form': self._calculate_player_form(player),
                    'importance': self._calculate_player_importance(player)
                })
        
        return sorted(key_players, key=lambda x: x['impact_score'], reverse=True)
    
    def _calculate_player_impact(self, player: Dict) -> float:
        """Calculate player's impact score"""
        metrics = {
            'goals': player.get('goals', 0) * 0.3,
            'assists': player.get('assists', 0) * 0.2,
            'minutes_played': player.get('minutes_played', 0) / 90 * 0.1,
            'pass_accuracy': player.get('pass_accuracy', 0) * 0.15,
            'duels_won': player.get('duels_won', 0) * 0.15,
            'recent_form': self._calculate_player_form(player) * 0.1
        }
        
        return sum(metrics.values())

    def _assess_squad_fatigue(self, player_stats: List[Dict]) -> Dict:
        """Assess squad fatigue levels"""
        fatigue_analysis = {
            'high_risk': [],
            'moderate_risk': [],
            'low_risk': [],
            'squad_fatigue_index': 0
        }
        
        for player in player_stats:
            fatigue_score = self._calculate_player_fatigue(player)
            
            if fatigue_score > 0.7:
                fatigue_analysis['high_risk'].append(player['name'])
            elif fatigue_score > 0.4:
                fatigue_analysis['moderate_risk'].append(player['name'])
            else:
                fatigue_analysis['low_risk'].append(player['name'])
                
            fatigue_analysis['squad_fatigue_index'] += fatigue_score
        
        fatigue_analysis['squad_fatigue_index'] /= len(player_stats)
        return fatigue_analysis
    
    def _calculate_player_fatigue(self, player: Dict) -> float:
        """Calculate individual player fatigue"""
        recent_minutes = player.get('recent_minutes', [])
        if not recent_minutes:
            return 0.0
            
        # Calculate acute to chronic workload ratio
        acute_load = sum(recent_minutes[-7:])  # Last 7 days
        chronic_load = sum(recent_minutes[-28:]) / 4  # 4-week average
        
        if chronic_load == 0:
            return 0.0
            
        acwr = acute_load / chronic_load
        return min(1.0, max(0.0, abs(acwr - 1.0)))

    def _analyze_squad_depth(self, player_stats: List[Dict]) -> Dict:
        """Analyze squad depth by position"""
        positions = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
        
        for player in player_stats:
            pos = player['position']
            if pos in positions:
                positions[pos].append({
                    'name': player['name'],
                    'rating': self._calculate_player_rating(player),
                    'form': self._calculate_player_form(player)
                })
        
        depth_analysis = {}
        for pos, players in positions.items():
            depth_analysis[pos] = {
                'num_players': len(players),
                'average_rating': np.mean([p['rating'] for p in players]),
                'depth_score': self._calculate_depth_score(players)
            }
        
        return depth_analysis
    
    def _calculate_depth_score(self, players: List[Dict]) -> float:
        """Calculate position depth score"""
        if not players:
            return 0.0
            
        ratings = sorted([p['rating'] for p in players], reverse=True)
        weighted_sum = sum(r * (0.9 ** i) for i, r in enumerate(ratings))
        return weighted_sum / len(players)