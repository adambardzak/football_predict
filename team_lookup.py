from typing import Dict
from venv import logger
from wsgiref import headers

import requests


def get_team_ids(league_id: int) -> Dict:
    """Fetch team IDs for a specific league"""
    endpoint = f"{API_FOOTBALL_BASE}/teams"
    params = {
        "league": league_id,
        "season": 2024
    }
    
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        teams = response.json()['response']
        
        return {team['team']['name']: team['team']['id'] for team in teams}
    except Exception as e:
        logger.error(f"Error fetching team IDs: {str(e)}")
        return {}

# Usage example:
premier_league_teams = get_team_ids(39)
la_liga_teams = get_team_ids(140)
serie_a_teams = get_team_ids(135)
czech_league_teams = get_team_ids(345)

print("Premier League Teams:", premier_league_teams)