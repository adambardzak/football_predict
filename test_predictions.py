import json
from script import FootballPredictor


def test_single_match():
    predictor = FootballPredictor(debug=True)
    
    # Test with Real Madrid vs Barcelona
    prediction = predictor.predict_match(
        home_team_id=541,  # Real Madrid
        away_team_id=529,  # Barcelona
        league_id=140,     # La Liga
        season=2023
    )
    
    print("\nTest Prediction Result:")
    print(json.dumps(prediction, indent=2))

if __name__ == "__main__":
    test_single_match()
