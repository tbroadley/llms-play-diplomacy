from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
import diplomacy
import time
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Define all powers in the game
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


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

Supply centers owned: {", ".join(power.centers)}
Current state of all units on the board:
{self._format_all_units()}

Game phase history:
{self._format_phase_history()}

Other powers' supply centers:
{self._format_supply_centers()}
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

    def _format_phase_history(self) -> str:
        """Format the phase history of the game."""
        return "\n".join([f"- {phase}" for phase in self.game.state_history])

    def _format_supply_centers(self) -> str:
        """Format the supply centers owned by each power."""
        centers = []
        for power in self.game.powers.values():
            centers.append(f"- {power.name}: {', '.join(power.centers)}")
        return "\n".join(centers)

    def submit_moves(self, moves_by_power: Dict[str, List[str]]) -> Dict[str, str]:
        """Submit moves for all powers and process the turn."""
        results = {}

        try:
            # Set the moves for each power
            for power_name, moves in moves_by_power.items():
                self.game.set_orders(power_name, moves)
                results[power_name] = f"Moves submitted for {power_name}: {moves}"

            # Process the turn
            self.game.process()

            return results
        except Exception as e:
            return {
                power: f"Error processing moves: {str(e)}"
                for power in moves_by_power.keys()
            }

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.game.is_game_done

    def get_winner(self) -> str:
        """Get the winner of the game if there is one."""
        if not self.is_game_over():
            return "Game not finished"

        for power_name in POWERS:
            power = self.game.get_power(power_name)
            if len(power.centers) >= 18:  # Win condition in Diplomacy
                return power_name
        return "No winner yet"


def get_moves_from_ai(game_state: str) -> List[str]:
    """Get moves from the AI based on the current game state."""

    system_message = """You are an AI playing Diplomacy. Your task is to suggest valid moves 
    in standard Diplomacy notation (e.g., 'F LON - NTH'). Analyze the game state carefully and 
    provide strategic moves. Consider supply centers owned and potential strategic positions.

    Use the submit_moves tool to submit your moves.

    Remember:
    - In Spring/Fall: Provide movement orders for all units
    - In Winter: If you need to build units, provide build orders (e.g., 'A LON B')
    - If you need to remove units, provide remove orders (e.g., 'A LON D')"""

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": game_state},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "submit_moves",
                    "description": "Submit the moves for the current power",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "moves": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        ],
    )

    tool_call = response.choices[0].message.tool_calls[0]
    if not tool_call or tool_call.function.name != "submit_moves":
        raise ValueError("No valid tool call found in the response")

    moves: list[str] = json.loads(tool_call.function.arguments)["moves"]
    return [move.strip() for move in moves if move.strip()]


def play_game(max_turns: int = 10, delay_between_turns: int = 2):
    """Play the game for a specified number of turns."""
    game = DiplomacyGame()
    turn_count = 0

    print("Starting new game with all powers...")
    print("-" * 50)

    while not game.is_game_over() and turn_count < max_turns:
        print(f"\nTurn {turn_count + 1} - {game.game.phase}")
        print("-" * 50)

        # Collect moves from all powers
        moves_by_power = {}
        for power_name in POWERS:
            print(f"\nGetting moves for {power_name}...")
            game_state = game.get_current_state(power_name)
            print(f"Current Game State for {power_name}:")
            print(game_state)

            moves = get_moves_from_ai(game_state)
            moves_by_power[power_name] = moves
            print(f"\n{power_name} Suggested Moves:")
            print("\n".join(moves))

        # Submit all moves and process the turn
        results = game.submit_moves(moves_by_power)
        print("\nResults:")
        for power, result in results.items():
            print(f"{power}: {result}")

        turn_count += 1

        # Add a delay between turns to make it easier to follow
        if turn_count < max_turns:
            print(f"\nWaiting {delay_between_turns} seconds before next turn...")
            time.sleep(delay_between_turns)

    print("\nGame finished!")
    print(f"Completed {turn_count} turns")
    if game.is_game_over():
        winner = game.get_winner()
        print(f"Game ended naturally. Winner: {winner}")
        # Print final supply center counts
        for power_name in POWERS:
            centers = len(game.game.get_power(power_name).centers)
            print(f"{power_name}: {centers} supply centers")
    else:
        print("Maximum turns reached")


if __name__ == "__main__":
    play_game(max_turns=10, delay_between_turns=2)
