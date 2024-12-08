from script import FootballPredictor
from visualization import PerformanceVisualization

class DailyPredictions:
    def __init__(self, predictor):
        self.predictor = predictor
        self.leagues = {
            'Premier League': 39,
            'La Liga': 140,
            'Serie A': 135,
            'Czech League': 345
        }
        
    
    def get_todays_matches(self):
        matches = self.predictor.data_collector.get_live_matches()
        return matches
    
    def predict_all_matches(self):
        matches = self.get_todays_matches()
        predictions = []
        
        for match in matches:
            try:
                home_team_id = match['teams']['home']['id']
                away_team_id = match['teams']['away']['id']
                league_id = match['league']['id']
                
                prediction = self.predictor.predict_match(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    league_id=league_id,
                    season=2023  # Current season
                )
                
                predictions.append({
                    'home_team': match['teams']['home']['name'],
                    'away_team': match['teams']['away']['name'],
                    'league': match['league']['name'],
                    'kickoff': match['fixture']['date'],
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'probabilities': prediction['probabilities']
                })
                
            except Exception as e:
                print(f"Error predicting match: {str(e)}")
                continue
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)

def main():
    predictor = FootballPredictor()
    daily_pred = DailyPredictions(predictor)
    
    print("Testing API connection...")
    predictor.data_collector.test_connection()
    
    print("\nFetching and predicting today's matches...")
    predictions = daily_pred.predict_all_matches()
    
    if not predictions:
        print("No matches found for today.")
        return
    
    print("\nMatch Predictions (sorted by confidence)")
    print("=====================================")
    
    for pred in predictions:
        print(f"\n{pred['league']}")
        print(f"{pred['home_team']} vs {pred['away_team']}")
        print(f"Prediction: {pred['prediction']}")
        print(f"Confidence: {pred['confidence']:.1%}")
        print(f"Kickoff: {pred['kickoff']}")
        print("-" * 40)
    
    # Create visualization
    print("\nGenerating visualization...")
    viz = PerformanceVisualization()
    confidence_plot = viz.create_prediction_confidence_plot(predictions)
    confidence_plot.show()

if __name__ == "__main__":
    main()