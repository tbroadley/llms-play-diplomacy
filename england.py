import os
import asyncio
import random
from typing import List, Dict, Optional
from dotenv import load_dotenv
from diplomacy import Game, Power
from diplomacy.utils.export import to_saved_game_format
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_functions

# Load environment variables (make sure to set OPENAI_API_KEY in .env)
load_dotenv()

class EnglandAgent:
    def __init__(self, game: Game):
        self.game = game
        self.power: Power = game.powers['ENGLAND']
        self.agent: Optional[AgentExecutor] = None

    @tool
    def submit_moves(self, moves: List[str]) -> str:
        """Submit a list of moves for England. Each move should be a valid Diplomacy order string.
        The moves should be in standard Diplomacy notation, for example:
        ["A LON H", "F NTH - NWY", "F ENG S A LON"]
        
        You must submit exactly one order for each unit England controls.
        All orders must be valid according to the current game state and Diplomacy rules.
        """
        if not isinstance(moves, list):
            return "Error: moves must be a list of strings"
        
        # Get current possible orders
        possible_orders = self.game.get_all_possible_orders()
        
        # Validate we have the right number of orders
        if len(moves) != len(self.power.units):
            return f"Error: Must submit exactly {len(self.power.units)} orders, got {len(moves)}"
        
        # Validate each move
        invalid_moves = []
        valid_moves = []
        
        for move in moves:
            # Extract the unit location from the order
            # Orders are typically in format "A LON - YOR" or "F NTH H" or "F ENG S A LON"
            try:
                unit_type = move[0]  # A or F
                location = move.split()[1]  # LON, NTH, etc.
                unit = f"{unit_type} {location}"
                
                # Check if we own this unit
                if unit not in self.power.units:
                    invalid_moves.append(f"{move} - We don't control unit {unit}")
                    continue
                
                # Check if this is a valid order for this unit
                if move not in possible_orders.get(location, []):
                    invalid_moves.append(f"{move} - Not a valid order for {unit}")
                    continue
                
                valid_moves.append(move)
            
            except (IndexError, KeyError):
                invalid_moves.append(f"{move} - Malformed order")
        
        if invalid_moves:
            return "Error - Invalid moves:\n" + "\n".join(invalid_moves)
        
        # Store valid moves for later use
        self.last_valid_moves = valid_moves
        return f"Success! Valid moves: {valid_moves}"

    async def create_agent(self):
        """Create the LangChain agent for England."""
        llm = ChatOpenAI(temperature=0.7)
        
        SYSTEM_PROMPT = """You are playing as England in a game of Diplomacy. Your goal is to expand your territory
        and eventually dominate Europe. You should analyze the current game state and make strategic moves for all your units.

        Key points to remember:
        1. Each unit (Army or Fleet) must receive exactly one order
        2. Orders must be valid according to Diplomacy rules
        3. Think strategically about supporting other units and coordinating movements
        4. Consider both offensive and defensive positions

        When submitting moves, use the standard Diplomacy notation:
        - A LON H (Army in London holds)
        - F NTH - NWY (Fleet in North Sea moves to Norway)
        - F ENG S A LON (Fleet in English Channel supports Army in London)

        You MUST use the submit_moves tool to validate and submit your orders.
        If the orders are invalid, analyze the error message and try again with corrected orders.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = [self.submit_moves]
        agent = create_openai_functions_agent(llm, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return self.agent

    async def get_orders(self) -> List[str]:
        """Get and validate orders from the LLM agent."""
        if not self.agent:
            await self.create_agent()
        
        # Prepare the game state information
        game_state = f"""
Current Phase: {self.game.phase}
England's Centers: {self.power.centers}
England's Units: {self.power.units}

Possible orders for each unit:
"""
        possible_orders = self.game.get_all_possible_orders()
        for unit in self.power.units:
            location = unit.split()[1]
            unit_orders = possible_orders.get(location, [])
            # Show first 5 possible orders as examples
            game_state += f"\n{unit}: {unit_orders[:5]}..."
        
        # Keep trying until we get valid orders
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Add attempt information if this isn't the first try
                if attempt > 0:
                    game_state += f"\n\nThis is attempt {attempt + 1} of {max_attempts}. Please ensure all orders are valid."
                
                result = await self.agent.ainvoke({"input": game_state})
                
                # If we got here and last_valid_moves exists, it means the submit_moves tool validated the orders
                if hasattr(self, 'last_valid_moves'):
                    return self.last_valid_moves
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_attempts - 1:
                    raise Exception("Failed to get valid orders after maximum attempts")
        
        raise Exception("Failed to get valid orders")

async def main():
    # Create a new game
    game = Game()
    england_agent = EnglandAgent(game)
    
    # Game loop
    while not game.is_game_done:
        # Get moves from the England LLM agent
        try:
            england_orders = await england_agent.get_orders()
            game.set_orders('ENGLAND', england_orders)
        except Exception as e:
            print(f"Error getting England's orders: {str(e)}")
            # If we fail to get valid orders, hold all units
            england_orders = [f"{unit} H" for unit in game.powers['ENGLAND'].units]
            game.set_orders('ENGLAND', england_orders)
        
        # Random moves for other powers
        possible_orders = game.get_all_possible_orders()
        for power_name, power in game.powers.items():
            if power_name != 'ENGLAND' and power.units:
                power_orders = []
                for unit in power.units:
                    location = unit.split()[1]
                    unit_possible_orders = possible_orders.get(location, [])
                    if unit_possible_orders:
                        power_orders.append(random.choice(unit_possible_orders))
                if power_orders:
                    game.set_orders(power_name, power_orders)
        
        # Process the turn
        game.process()
        print(f"\nProcessed {game.phase} - {game.get_state()['name']}")
        
        # Print the current state of each power
        for power_name, power in game.powers.items():
            print(f"{power_name}: {len(power.centers)} centers, {len(power.units)} units")
            if power.orders:
                print(f"  Orders: {power.orders}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 