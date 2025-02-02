import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def submit_moves(moves: List[str]) -> str:
    """Submit moves for the current turn. Each move should be a valid Diplomacy order."""
    # In a real implementation, this would interact with the game
    return f"Submitted moves: {moves}"

def get_moves_from_ai(game_state: str) -> List[str]:
    """Get moves from the AI based on the current game state."""
    
    system_message = """You are an AI playing Diplomacy. Your task is to suggest valid moves 
    in standard Diplomacy notation (e.g., 'F LON - NTH'). Provide exactly three moves, one per line."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": game_state}
        ]
    )
    
    # Extract moves from the response
    moves = response.choices[0].message.content.strip().split('\n')
    return [move.strip() for move in moves if move.strip()]

if __name__ == "__main__":
    # Example game state
    game_state = """
    You are playing as England in Spring 1901. You control:
    - Fleet London (F LON)
    - Fleet Edinburgh (F EDI)
    - Army Liverpool (A LVP)
    
    Provide three opening moves in standard Diplomacy notation.
    """
    
    # Get moves from AI
    moves = get_moves_from_ai(game_state)
    
    # Submit the moves
    result = submit_moves(moves)
    print(result) 