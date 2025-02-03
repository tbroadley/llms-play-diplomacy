from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import List, Dict
import diplomacy
import json
import asyncio
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI()

# Define all powers in the game
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


class DiplomacyGame:
    def __init__(self):
        """Initialize a new standard Diplomacy game."""
        self.game = diplomacy.Game()
        self.turn_start_time = None

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

    def has_moves_to_make(self, power: str) -> bool:
        """Check if a power has moves to make."""
        power = self.game.get_power(power)
        if self.game.phase_type == "R":
            return len(power.retreats.keys()) > 0
        if self.game.phase_type == "A":
            return len(power.centers) != len(power.units)
        return not power.moves_submitted()

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


async def get_moves_from_ai(game_state: str, power_name: str) -> List[str]:
    """Get moves from the AI based on the current game state."""

    system_message = f"""
You are an AI playing as {power_name} in the game of Diplomacy. Your task is to suggest valid moves using standard Diplomacy notation. Analyze the game state carefully and provide strategic moves, considering supply centers owned and potential strategic positions.

Use the `submit_moves` tool to submit your moves.

### Movement Phase Orders:
- **Movement Orders**: In Spring and Fall, provide movement orders for all units. Use the format `UnitType Location - Destination` (e.g., `F LON - NTH` for a fleet moving from London to the North Sea).
- **Hold Orders**: To hold a position, use the format `UnitType Location H` (e.g., `A PAR H` to hold an army in Paris).
- **Support Orders**: To support another unit's move, use the format `UnitType Location S UnitType Location - Destination` (e.g., `A PAR S A BUR - MUN` to support an army in Burgundy moving to Munich from Paris).
- **Convoy Orders**: To convoy an army across water, use the format `F Location C A Location - Destination` (e.g., `F ENG C A LON - BRE` to convoy an army from London to Brest via the English Channel).

### Retreat Phase Orders:
- **Retreat Orders**: To retreat, use the format `UnitType Location R Destination` (e.g., `A PAR R BUR` to retreat an army from Paris to Burgundy).
- **Disband Orders**: If you want to disband a unit instead of retreating, use the format `UnitType Location D` (e.g., `A PAR D` to disband an army in Paris).

### Adjustment Phase Orders:
- **Build Orders**: In Winter, if you need to build units, use the format `UnitType Location B` (e.g., `A LON B` to build an army in London).
- **Waiving a Build**: To waive a build, use the order `WAIVE`. Use this order multiple times if you need to waive multiple builds.
- **Disband Orders**: If you need to remove units, use the format `UnitType Location D` (e.g., `A LON D` to disband an army in London).

### Strategic Considerations:
- **Supply Centers**: Focus on capturing and holding supply centers to build more units.
- **Alliances and Conflicts**: Consider potential alliances and conflicts with other powers. Support and convoy orders can be crucial in forming strategic partnerships.
- **Defense and Expansion**: Balance between defending your current territories and expanding into new ones.

### Submission:
- Use the `submit_moves` tool to submit your moves once you have determined the best strategy.
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
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
    except Exception as e:
        print(f"Error getting moves for {power_name}: {str(e)}")
        return []


async def play_game(max_turns: int):
    """Play the game for a specified number of turns."""
    game = DiplomacyGame()
    renderer = diplomacy.engine.renderer.Renderer(game.game)
    turn_count = 0

    output_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting new game with all powers...")
    print("-" * 50)

    while not game.is_game_over() and turn_count < max_turns:
        turn_start_time = datetime.now()
        print(f"\nTurn {turn_count + 1} - {game.game.phase}")
        print("-" * 50)

        # Collect moves from all powers in parallel
        moves_by_power = {}
        tasks = []

        # Create tasks for all powers
        for power_name in POWERS:
            if not game.has_moves_to_make(power_name):
                continue

            print(f"Preparing moves for {power_name}...")
            game_state = game.get_current_state(power_name)
            task = asyncio.create_task(get_moves_from_ai(game_state, power_name))
            tasks.append((power_name, task))

        # Wait for all AI responses in parallel
        for power_name, task in tasks:
            moves = await task
            moves_by_power[power_name] = moves
            print(f"\n{power_name} Suggested Moves:")
            print("\n".join(moves))

        # Submit all moves and process the turn
        results = game.submit_moves(moves_by_power)
        print("\nResults:")
        for power, result in results.items():
            print(f"{power}: {result}")

        turn_count += 1
        turn_duration = datetime.now() - turn_start_time
        print(
            f"\nTurn {turn_count} completed in {turn_duration.total_seconds():.2f} seconds"
        )

        renderer.render(output_path=output_dir / "game.svg", incl_orders=True)
        renderer.render(
            output_path=output_dir / f"{game.game.phase}.svg", incl_orders=True
        )
        diplomacy.utils.export.to_saved_game_format(
            game.game, output_dir / "game_states.jsonl"
        )

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
    asyncio.run(play_game(max_turns=100))
