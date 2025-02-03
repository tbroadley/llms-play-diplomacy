from pathlib import Path
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from dotenv import load_dotenv
from typing import List, Dict, Literal, Tuple
import diplomacy
import json
import asyncio
from datetime import datetime

from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI()

# Define all powers in the game
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


class Message(BaseModel):
    power: str
    message: str


class DiplomacyGame:
    def __init__(self):
        """Initialize a new standard Diplomacy game."""
        self.game = diplomacy.Game()
        self.turn_start_time = None
        self.public_messages: List[Message] = []
        self.private_messages: Dict[Tuple[str, str], Message] = {}

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

Public messages:
{self._format_public_messages()}

Private messages:
{self._format_private_messages(power_name)}
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

    def _format_public_messages(self) -> str:
        """Format the public messages of the game."""
        return "\n".join(
            [
                f"- {message.power}: {message.message}"
                for message in self.public_messages
            ]
        )

    def _format_private_messages(self, power_name: str) -> str:
        """Format the private messages of the game."""
        conversations = {
            (p1 if p2 == power_name else p2): messages
            for (p1, p2), messages in self.private_messages.items()
            if p1 == power_name or p2 == power_name
        }
        return "\n\n".join(
            [
                f"Conversation with {power}:\n{messages}"
                for power, messages in conversations.items()
            ]
        )

    def has_moves_to_make(self, power: str) -> bool:
        """Check if a power has moves to make."""
        power = self.game.get_power(power)
        if self.game.phase_type == "R":
            return len(power.retreats.keys()) > 0
        if self.game.phase_type == "A":
            return len(power.centers) != len(power.units)
        return not power.moves_submitted()

    def submit_moves(self, power_name: str, moves: List[str]) -> str:
        """Submit moves for a power and process the turn."""
        try:
            self.game.set_orders(power_name, moves)
            return f"Moves submitted for {power_name}: {moves}"
        except Exception as e:
            return f"Error processing moves: {str(e)}"

    def send_public_message(self, power_name: str, message: str) -> str:
        """Send a public message to all powers."""
        self.public_messages.append(Message(power=power_name, message=message))
        return f"Public message sent to all powers: {message}"

    def send_private_message(
        self, power_name: str, message: str, target_power: str
    ) -> str:
        """Send a private message to a specific power."""
        key = (
            (power_name, target_power)
            if power_name < target_power
            else (target_power, power_name)
        )
        self.private_messages[key].append(Message(power=power_name, message=message))
        return f"Private message sent to {target_power}: {message}"

    def process_turn(self) -> str:
        """Process the turn."""
        self.game.process()

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


class SubmitMovesAction(BaseModel):
    type: Literal["submit_moves"]
    moves: List[str]


class SendPublicMessageAction(BaseModel):
    type: Literal["send_public_message"]
    message: str


class SendPrivateMessageAction(BaseModel):
    type: Literal["send_private_message"]
    message: str
    power: str


class SleepAction(BaseModel):
    type: Literal["sleep"]
    seconds: int


Action = (
    SubmitMovesAction | SendPublicMessageAction | SendPrivateMessageAction | SleepAction
)


class Player:
    def __init__(self, game: DiplomacyGame, power_name: str):
        self.game = game
        self.power_name = power_name

    async def _get_actions(self, game_state: str) -> List[Action]:
        """Get moves from the AI based on the current game state."""

        system_message = f"""
    You are an AI playing as {self.power_name} in the game of Diplomacy. Your task is to suggest valid moves using standard Diplomacy notation. Analyze the game state carefully and provide strategic moves, considering supply centers owned and potential strategic positions.

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

    ### Tools:
    - `send_public_message`: Send a public message to all powers.
    - `send_private_message`: Send a private message to a specific power.
    - `sleep`: Do nothing for a specified number of seconds.
    - `submit_moves`: Submit the moves for the current power.
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
                            "name": "send_public_message",
                            "description": "Send a public message to all powers",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                },
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "send_private_message",
                            "description": "Send a private message to a specific power",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "power": {"type": "string", "enum": POWERS},
                                },
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "submit_moves",
                            "description": "Submit moves for your power",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "moves": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "sleep",
                            "description": "Do nothing for a specified number of seconds",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "seconds": {"type": "integer"},
                                },
                            },
                        },
                    },
                ],
            )

            return [
                self._parse_tool_call(tool_call)
                for tool_call in response.choices[0].message.tool_calls or []
            ]
        except Exception as e:
            print(f"Error getting actions for {self.power_name}: {str(e)}")
            return []

    def _parse_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> Action:
        arguments = json.loads(tool_call.function.arguments)
        if tool_call.function.name == "submit_moves":
            return SubmitMovesAction(**arguments)
        elif tool_call.function.name == "send_public_message":
            return SendPublicMessageAction(**arguments)
        elif tool_call.function.name == "send_private_message":
            return SendPrivateMessageAction(**arguments)
        elif tool_call.function.name == "sleep":
            return SleepAction(**arguments)
        return None

    async def run(self):
        while True:
            game_state = self.game.get_current_state(self.power_name)
            actions = await self._get_actions(game_state)
            for action in actions:
                if action.type == "submit_moves":
                    self.game.submit_moves(self.power_name, action.moves)
                elif action.type == "send_public_message":
                    self.game.send_public_message(self.power_name, action.message)
                elif action.type == "send_private_message":
                    self.game.send_private_message(
                        self.power_name, action.message, action.power
                    )
                elif action.type == "sleep":
                    await asyncio.sleep(action.seconds)


async def play_game(turn_time_limit: int, max_turns: int):
    """Play the game for a specified number of turns."""
    game = DiplomacyGame()
    renderer = diplomacy.engine.renderer.Renderer(game.game)
    turn_count = 0

    output_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting new game with all powers...")
    print("-" * 50)

    players = {power_name: Player(power_name) for power_name in POWERS}

    while not game.is_game_over() and turn_count < max_turns:
        turn_start_time = datetime.now()
        print(f"\nTurn {turn_count + 1} - {game.game.phase}")
        print("-" * 50)

        # Collect moves from all powers in parallel
        tasks = []

        # Create tasks for all powers
        for power_name, player in players.items():
            if not game.has_moves_to_make(power_name):
                continue

            print(f"Running {power_name}...")
            task = asyncio.create_task(player.run())
            tasks.append(task)

        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=turn_time_limit)
        except asyncio.TimeoutError:
            for _, task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Process the turn
        game.process_turn()

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
