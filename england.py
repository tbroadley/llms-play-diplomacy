import os
import asyncio
import random
from typing import List, Dict, Optional
from dotenv import load_dotenv
from diplomacy import Game, Power
from diplomacy.utils.export import to_saved_game_format
from langchain_core.tools import tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_functions
from langchain.callbacks import get_openai_callback

# Load environment variables (make sure to set OPENAI_API_KEY in .env)
load_dotenv()

class EnglandAgent:
    def __init__(self, game: Game):
        self.game = game
        self.power: Power = game.powers['ENGLAND']
        self.agent: Optional[AgentExecutor] = None
        self.chat_history = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def submit_moves(self, moves: List[str]) -> str:
        """Submit a list of moves for England. Each move should be a valid Diplomacy order string.
        The moves should be in standard Diplomacy notation, for example:
        ["A LON H", "F NTH - NWY", "F ENG S A LON"]
        
        For Movement phases:
        - Submit exactly one order for each unit
        For Retreat phases:
        - Submit exactly one order for each dislodged unit
        For Adjustment phases:
        - Submit build/disband orders based on supply center count
        """
        if not isinstance(moves, list):
            return "Error: moves must be a list of strings"
        
        phase_type = self.game.phase_type
        
        # Validate number of orders based on phase type
        if phase_type == 'M':
            if len(moves) != len(self.power.units):
                return f"Error: Must submit exactly {len(self.power.units)} orders in movement phase, got {len(moves)}"
        elif phase_type == 'R':
            dislodged_units = [unit for unit in self.power.units if unit.startswith('*')]
            if len(moves) != len(dislodged_units):
                return f"Error: Must submit exactly {len(dislodged_units)} orders in retreat phase, got {len(moves)}"
        elif phase_type == 'A':
            # For adjustment phase, number of orders depends on the difference between centers and units
            center_delta = len(self.power.centers) - len(self.power.units)
            if center_delta > 0:
                # Can build units
                if len(moves) != center_delta:
                    return f"Error: Must submit exactly {center_delta} build orders, got {len(moves)}"
            elif center_delta < 0:
                # Must remove units
                if len(moves) != abs(center_delta):
                    return f"Error: Must submit exactly {abs(center_delta)} disband orders, got {len(moves)}"
            else:
                # No orders needed
                if len(moves) != 0:
                    return "Error: No orders needed in adjustment phase when centers equal units"
        
        # Get current possible orders
        possible_orders = self.game.get_all_possible_orders()
        
        # Validate each move
        invalid_moves = []
        valid_moves = []
        
        for move in moves:
            try:
                # Extract the unit location from the order
                if phase_type == 'A' and 'B' in move:
                    # Build order
                    location = move.split()[1]
                    if move not in possible_orders.get(location, []):
                        invalid_moves.append(f"{move} - Not a valid build order")
                        continue
                else:
                    # Movement, retreat, or disband order
                    unit_type = move[0]  # A or F
                    location = move.split()[1]  # LON, NTH, etc.
                    unit = f"{unit_type} {location}"
                    
                    # For non-build orders, check if we own the unit
                    if phase_type != 'A' and unit not in self.power.units and f"*{unit}" not in self.power.units:
                        invalid_moves.append(f"{move} - We don't control unit {unit}")
                        continue
                    
                    # Check if this is a valid order
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
        
        # Get map information
        map_info = {}
        for loc in self.game.map.abuts:
            # Only include land and sea territories, skip coast specifications
            if not any(c in loc for c in ['/', '-']):
                neighbors = set()
                for unit_type in self.game.map.abuts[loc]:
                    neighbors.update(n.split('/')[0] for n in self.game.map.abuts[loc][unit_type])
                map_info[loc] = sorted(list(neighbors))
        
        SYSTEM_PROMPT = f"""You are playing as England in a game of Diplomacy. Your goal is to expand your territory
        and eventually dominate Europe. You should analyze the current game state and make strategic moves for all your units.

        Key points to remember:
        1. Orders must be valid according to Diplomacy rules
        2. Think strategically about supporting other units and coordinating movements
        3. Consider both offensive and defensive positions
        4. Pay attention to the current phase type:
           - MOVEMENT (M): Regular movement orders, supports, and convoys. Submit one order per unit.
           - RETREAT (R): Units that were dislodged must retreat or be disbanded. Submit one order per dislodged unit.
           - ADJUSTMENT (A): Build new units in home centers or remove existing units. Number of orders depends on supply center count.

        When submitting moves, use the standard Diplomacy notation:
        For Movement phases:
        - A LON H (Army in London holds)
        - F NTH - NWY (Fleet in North Sea moves to Norway)
        - F ENG S A LON (Fleet in English Channel supports Army in London)
        - F NTH C A YOR - NWY (Fleet in North Sea convoys Army in Yorkshire to Norway)

        For Retreat phases:
        - A LON - YOR (Army in London retreats to Yorkshire)
        - A LON D (Army in London disbands)

        For Adjustment phases:
        - A LON B (Build Army in London)
        - F EDI B (Build Fleet in Edinburgh)
        - A LON D (Disband Army in London)

        Map Information (territory: [adjacent territories]):
        {chr(10).join(f"{loc}: {neighbors}" for loc, neighbors in map_info.items())}

        You MUST use the submit_moves tool to validate and submit your orders.
        If the orders are invalid, analyze the error message and try again with corrected orders.
        Submit your moves as a list of strings, e.g.: ["A LON H", "F NTH - NWY"]
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create a properly bound tool
        submit_moves_tool = StructuredTool.from_function(
            func=self.submit_moves,
            name="submit_moves",
            description=self.submit_moves.__doc__
        )
        
        tools = [submit_moves_tool]
        agent = create_openai_functions_agent(llm, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return self.agent

    async def get_orders(self) -> List[str]:
        """Get and validate orders from the LLM agent."""
        if not self.agent:
            await self.create_agent()
        
        # Clear any previous valid moves
        if hasattr(self, 'last_valid_moves'):
            delattr(self, 'last_valid_moves')
        
        # Get the current game state
        state = self.game.get_state()
        current_phase = self.game.get_current_phase()
        phase_type = self.game.phase_type
        
        # Prepare the game state information
        game_state = f"""
Current Game State:
Phase: {current_phase} ({phase_type})
Year: {state['name']}

England's Status:
Centers: {sorted(list(self.power.centers))}
Units: {sorted(list(self.power.units))}
Supply Centers Needed: {self.power.influence}

Other Powers' Status:
{chr(10).join(f"{power}: {len(centers)} centers, {len(self.game.powers[power].units)} units"
              for power, centers in self.game.get_centers().items() if power != 'ENGLAND')}

Instructions:
1. This is a {phase_type} phase. {
    "Submit one movement order for each unit." if phase_type == "M" else
    "Submit one retreat order for each dislodged unit." if phase_type == "R" else
    f"{'Build' if len(self.power.centers) > len(self.power.units) else 'Disband'} {abs(len(self.power.centers) - len(self.power.units))} units." if phase_type == "A" else
    "Unknown phase type"
}
2. Orders must be valid according to the current phase and game rules.
"""
        
        # Keep trying until we get valid orders
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Add attempt information if this isn't the first try
                if attempt > 0:
                    game_state += f"\n\nThis is attempt {attempt + 1} of {max_attempts}. Previous attempts failed. Please ensure all orders are valid."
                
                # Track costs for this attempt
                with get_openai_callback() as cb:
                    result = await self.agent.ainvoke({
                        "input": game_state,
                        "chat_history": self.chat_history
                    })
                    self.total_cost += cb.total_cost
                    self.total_tokens += cb.total_tokens
                    print(f"\nLLM Usage - Total Cost: ${self.total_cost:.4f}, Total Tokens: {self.total_tokens}")
                
                # If we got here and last_valid_moves exists, it means the submit_moves tool validated the orders
                if hasattr(self, 'last_valid_moves'):
                    # Add the successful orders to chat history
                    self.chat_history.append(("human", game_state))
                    self.chat_history.append(("assistant", f"Orders submitted: {self.last_valid_moves}"))
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
    while not game.is_game_done and int(game.get_current_phase()[1:5]) <= 1901:
        # Get moves from the England LLM agent
        try:
            england_orders = await england_agent.get_orders()
            game.set_orders('ENGLAND', england_orders)
        except Exception as e:
            print(f"Error getting England's orders: {str(e)}")
            # If we fail to get valid orders, hold all units in movement phase
            # or disband units in retreat/adjustment phase
            if game.phase_type == 'M':
                england_orders = [f"{unit} H" for unit in game.powers['ENGLAND'].units]
            else:
                england_orders = [f"{unit} D" for unit in game.powers['ENGLAND'].units]
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