from typing import Dict, List, Tuple
import diplomacy
from collections import defaultdict
from pydantic import BaseModel


class Message(BaseModel):
    power: str
    message: str


class DiplomacyGame:
    def __init__(self, turn_time_limit: int):
        """Initialize a new standard Diplomacy game."""
        self.game = diplomacy.Game()
        self.turn_time_limit = turn_time_limit
        self.public_messages: List[Message] = []
        self.private_messages: Dict[Tuple[str, str], List[Message]] = defaultdict(list)

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
                all_units.append(
                    f"- {power.name}: {unit} (can move to {', '.join(self.game.map.abut_list(unit.split(' ')[1]))})"
                )
        return "\n".join(all_units)

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
        return self.game.note
