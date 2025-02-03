from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from typing import List
import json
import asyncio
from datetime import datetime, timedelta
from pydantic import BaseModel
from game import DiplomacyGame

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


class SubmitMovesAction(BaseModel):
    moves: List[str]


class SendPublicMessageAction(BaseModel):
    message: str


class SendPrivateMessageAction(BaseModel):
    message: str
    power: str


class SleepAction(BaseModel):
    seconds: int


Action = (
    SubmitMovesAction | SendPublicMessageAction | SendPrivateMessageAction | SleepAction
)


class Player:
    def __init__(self, game: DiplomacyGame, power_name: str, client: AsyncOpenAI):
        self.game = game
        self.power_name = power_name
        self.client = client

    async def _get_actions(self, game_state: str, end_time: datetime) -> List[Action]:
        """Get moves from the AI based on the current game state."""

        system_message = f"""
    You are an AI playing as {self.power_name} in the game of Diplomacy.

    ### Communication:
    - You can send public messages to all powers using the `send_public_message` tool.
    - You can send private messages to specific powers using the `send_private_message` tool.
    - You can sleep for a specified number of seconds using the `sleep` tool, to wait for other powers to send you messages.

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

    ### Time:
    You have {end_time - datetime.now()} left to submit your moves.
    """

        try:
            response = await self.client.chat.completions.create(
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
        end_time = datetime.now() + timedelta(seconds=self.game.turn_time_limit)

        while True:
            game_state = self.game.get_current_state(self.power_name)
            actions = await self._get_actions(game_state, end_time)
            print(f"Actions for {self.power_name}: {actions}")
            for action in actions:
                if isinstance(action, SubmitMovesAction):
                    self.game.submit_moves(self.power_name, action.moves)
                elif isinstance(action, SendPublicMessageAction):
                    self.game.send_public_message(self.power_name, action.message)
                elif isinstance(action, SendPrivateMessageAction):
                    self.game.send_private_message(
                        self.power_name, action.message, action.power
                    )
                elif isinstance(action, SleepAction):
                    await asyncio.sleep(action.seconds)
