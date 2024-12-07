from script import FootballPredictor
from simulation import DailyPredictions
from visualization import PerformanceVisualization
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv('RAPIDAPI_KEY'):
        print("Please set your RAPIDAPI_KEY in the .env file")
        return

    try:
        # Initialize predictor
        predictor = FootballPredictor()
        
        # Example: Premier League match
        print("\nPredicting upcoming matches...")
        
        # League IDs
        LEAGUES = {
            'Premier League': 39,
            'La Liga': 140,
            'Serie A': 135,
            'Czech League': 345
        }
        
        # Create daily predictions instance
        daily_pred = DailyPredictions(predictor, leagues=list(LEAGUES.values()))
        
        # Get and display today's predictions
        predictions = daily_pred.predict_daily_matches()
        daily_pred.display_predictions(predictions)
        
        # Create visualization
        print("\nGenerating visualization...")
        viz = PerformanceVisualization()
        confidence_plot = viz.create_prediction_confidence_plot(predictions)
        confidence_plot.show()
        
    except Exception as e:
        logger.error(f"Error in main program: {str(e)}")
        print(f"An error occurred: {str(e)}")
        print("Check your API key and internet connection")

if __name__ == "__main__":
    main()