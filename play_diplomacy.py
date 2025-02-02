import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import diplomacy

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

class DiplomacyGame:
    def __init__(self):
        """Initialize a new standard Diplomacy game."""
        self.game = diplomacy.Game()
        
    def get_current_state(self, power_name: str) -> str:
        """Get the current game state for a specific power."""
        power = self.game.get_power(power_name)
        units = power.units
        
        # Format the game state information
        state = f"""
You are playing as {power_name} in {self.game.phase}.
You control the following units:
{self._format_units(units)}

Current state of all units on the board:
{self._format_all_units()}
"""
        return state
    
    def _format_units(self, units: List[str]) -> str:
        """Format a list of units into a readable string."""
        return "\n".join([f"- {unit}" for unit in units])
    
    def _format_all_units(self) -> str:
        """Format all units on the board into a readable string."""
        all_units = []
        for power in self.game.powers.values():
            for unit in power.units:
                all_units.append(f"- {power.name}: {unit}")
        return "\n".join(all_units)
    
    def submit_moves(self, power_name: str, moves: List[str]) -> str:
        """Submit moves for a power and process the turn."""
        try:
            # Set the moves for the power
            self.game.set_orders(power_name, moves)
            
            # Process the turn
            self.game.process()
            
            return f"Moves submitted and processed successfully for {power_name}: {moves}"
        except Exception as e:
            return f"Error processing moves: {str(e)}"

def get_moves_from_ai(game_state: str) -> List[str]:
    """Get moves from the AI based on the current game state."""
    
    system_message = """You are an AI playing Diplomacy. Your task is to suggest valid moves 
    in standard Diplomacy notation (e.g., 'F LON - NTH'). Analyze the game state carefully and 
    provide strategic moves. Each move should be on a new line and should be valid for the current
    game state. Consider adjacent territories and potential strategic positions."""
    
    response = client.chat.completions.create(
        model="gpt-4",
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
    # Create a new game instance
    game = DiplomacyGame()
    power_name = "ENGLAND"
    
    # Get the current game state
    game_state = game.get_current_state(power_name)
    print("Current Game State:")
    print(game_state)
    
    # Get moves from AI
    moves = get_moves_from_ai(game_state)
    print("\nAI Suggested Moves:")
    print("\n".join(moves))
    
    # Submit the moves and process the turn
    result = game.submit_moves(power_name, moves)
    print("\nResult:")
    print(result)
    
    # Show updated game state
    print("\nUpdated Game State:")
    print(game.get_current_state(power_name)) 