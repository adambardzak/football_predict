from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Football Match Prediction API",
    description="API for predicting football match outcomes and providing detailed analysis",
    version="1.0.0"
)

class MatchPredictionRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    league_id: int
    season: int
    include_simulation: Optional[bool] = False
    include_player_analysis: Optional[bool] = False
    include_betting_analysis: Optional[bool] = False

class MatchPredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    confidence_score: float
    simulation_results: Optional[Dict] = None
    player_analysis: Optional[Dict] = None
    betting_analysis: Optional[Dict] = None

@app.post("/predict", response_model=MatchPredictionResponse)
async def predict_match(request: MatchPredictionRequest):
    try:
        # Initialize predictor
        predictor = FootballPredictor()
        
        # Get base prediction
        prediction = predictor.predict_match(
            request.home_team_id,
            request.away_team_id,
            request.league_id,
            request.season
        )
        
        response = {
            "prediction": prediction["prediction"],
            "probabilities": prediction["probabilities"],
            "confidence_score": prediction["confidence_score"]
        }
        
        # Add simulation results if requested
        if request.include_simulation:
            simulator = MatchSimulation(predictor)
            simulation_results = simulator.simulate_match(
                request.home_team_id,
                request.away_team_id,
                request.league_id,
                request.season
            )
            response["simulation_results"] = simulation_results
        
        # Add player analysis if requested
        if request.include_player_analysis:
            player_analyzer = AdvancedPlayerAnalysis()
            response["player_analysis"] = {
                "home_team": player_analyzer.analyze_player_performance(request.home_team_id),
                "away_team": player_analyzer.analyze_player_performance(request.away_team_id)
            }
        
        # Add betting analysis if requested
        if request.include_betting_analysis:
            betting_analyzer = BettingIntegration()
            odds = betting_analyzer.get_match_odds(request.match_id)
            value_bets = betting_analyzer.calculate_value_bets(
                prediction["probabilities"],
                odds
            )
            response["betting_analysis"] = {
                "odds": odds,
                "value_bets": value_bets
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)