# visualization.py
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PerformanceVisualization:
    """Create visualizations for match and prediction analysis"""
    
    def __init__(self):
        self.LEAGUE_COLORS = {
            39: '#3d195b',   # Premier League (purple)
            140: '#ff3939',  # La Liga (red)
            135: '#008fd7',  # Serie A (blue)
            345: '#e31b23'   # Czech First League (red)
        }
    
    def create_prediction_confidence_plot(self, predictions: List[Dict]) -> go.Figure:
        """Create visualization of prediction confidence levels"""
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        fig = go.Figure()
        
        # Add bars for each match prediction
        fig.add_trace(go.Bar(
            x=[f"{p['home_team']} vs {p['away_team']}" for p in predictions],
            y=[p['confidence'] * 100 for p in predictions],
            text=[f"{p['prediction']} ({p['confidence']:.1%})" for p in predictions],
            textposition='auto',
            marker_color=[self.LEAGUE_COLORS.get(p['league_id'], '#808080') 
                         for p in predictions]
        ))
        
        fig.update_layout(
            title='Match Predictions Confidence Levels',
            xaxis_title='Matches',
            yaxis_title='Confidence (%)',
            template='plotly_white'
        )
        
        return fig
    
    def create_league_performance_matrix(self, predictions: List[Dict]) -> go.Figure:
        """Create matrix showing prediction performance by league"""
        league_data = {}
        
        for pred in predictions:
            league_id = pred['league_id']
            if league_id not in league_data:
                league_data[league_id] = {
                    'correct': 0,
                    'total': 0,
                    'avg_confidence': 0
                }
            
            league_data[league_id]['total'] += 1
            if pred.get('actual_result') == pred['prediction']:
                league_data[league_id]['correct'] += 1
            league_data[league_id]['avg_confidence'] += pred['confidence']
        
        # Calculate averages
        for league in league_data.values():
            league['accuracy'] = league['correct'] / league['total']
            league['avg_confidence'] = league['avg_confidence'] / league['total']
        
        return self._create_league_matrix_plot(league_data)
    
    def _create_league_matrix_plot(self, league_data: Dict) -> go.Figure:
        """Create the league performance matrix visualization"""
        fig = go.Figure()
        
        for league_id, data in league_data.items():
            fig.add_trace(go.Scatter(
                x=[data['avg_confidence']],
                y=[data['accuracy']],
                mode='markers+text',
                name=f"League {league_id}",
                marker=dict(
                    size=20,
                    color=self.LEAGUE_COLORS.get(league_id, '#808080')
                ),
                text=[f"League {league_id}"],
                textposition="top center"
            ))
        
        fig.update_layout(
            title='League Prediction Performance Matrix',
            xaxis_title='Average Confidence',
            yaxis_title='Actual Accuracy',
            template='plotly_white'
        )
        
        return fig