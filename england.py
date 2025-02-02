import asyncio
import random
from typing import List, Optional
from dotenv import load_dotenv
from diplomacy import Game, Power
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks.manager import get_openai_callback

# Load environment variables from .env (ensure your OPENAI_API_KEY is set)
load_dotenv()

class EnglandAgent:
    """
    A LangChain-based agent for generating and validating orders for England in a Diplomacy game.
    
    Attributes:
        game (Game): The game instance.
        power (Power): The "ENGLAND" power from the game.
        agent (Optional[AgentExecutor]): The LangChain agent executor.
        chat_history (List): The chat history maintained for the agent.
    """
    
    def __init__(self, game: Game):
        self.game = game
        self.power: Power = game.powers['ENGLAND']
        self.agent: Optional[AgentExecutor] = None
        self.chat_history = []
    
    def submit_moves(self, moves: List[str]) -> str:
        """
        Validates and submits a list of moves for England.
        
        Each move must be a standard Diplomacy order. Validation rules depend on the current phase:
          - MOVEMENT (M): Exactly one order per unit.
          - RETREAT (R): Exactly one order per dislodged unit.
          - ADJUSTMENT (A): Build/disband orders based on the difference between centers and units.
        
        Returns:
            A success message with validated moves, or an error message if any move is invalid.
        """
        if not isinstance(moves, list):
            return "Error: moves must be a list of strings"
        
        phase_type = self.game.phase_type
        
        # Validate order count based on phase type
        if phase_type == 'M':
            if len(moves) != len(self.power.units):
                return f"Error: Must submit exactly {len(self.power.units)} orders in movement phase, got {len(moves)}"
        elif phase_type == 'R':
            dislodged_units = [unit for unit in self.power.units if unit.startswith('*')]
            if len(moves) != len(dislodged_units):
                return f"Error: Must submit exactly {len(dislodged_units)} orders in retreat phase, got {len(moves)}"
        elif phase_type == 'A':
            center_delta = len(self.power.centers) - len(self.power.units)
            if center_delta > 0:
                if len(moves) != center_delta:
                    return f"Error: Must submit exactly {center_delta} build orders, got {len(moves)}"
            elif center_delta < 0:
                if len(moves) != abs(center_delta):
                    return f"Error: Must submit exactly {abs(center_delta)} disband orders, got {len(moves)}"
            else:
                if len(moves) != 0:
                    return "Error: No orders needed in adjustment phase when centers equal units"
        
        # Validate each move against possible orders from the game state
        possible_orders = self.game.get_all_possible_orders()
        invalid_moves = []
        valid_moves = []
        
        for move in moves:
            try:
                if phase_type == 'A' and 'B' in move:
                    # Build order
                    location = move.split()[1]
                    if move not in possible_orders.get(location, []):
                        invalid_moves.append(f"{move} - Not a valid build order")
                        continue
                else:
                    # Movement, retreat, or disband order
                    unit_type = move[0]  # 'A' or 'F'
                    location = move.split()[1]
                    unit = f"{unit_type} {location}"
                    
                    # For non-build orders, ensure control over the unit
                    if phase_type != 'A' and unit not in self.power.units and f"*{unit}" not in self.power.units:
                        invalid_moves.append(f"{move} - We don't control unit {unit}")
                        continue
                    
                    if move not in possible_orders.get(location, []):
                        invalid_moves.append(f"{move} - Not a valid order for {unit}")
                        continue
                valid_moves.append(move)
            except (IndexError, KeyError):
                invalid_moves.append(f"{move} - Malformed order")
        
        if invalid_moves:
            return "Error - Invalid moves:\n" + "\n".join(invalid_moves)
        
        # Save the valid moves for later use by the agent
        self.last_valid_moves = valid_moves
        return f"Success! Valid moves: {valid_moves}"
    
    def _build_game_state(self, phase_type: str, state: dict) -> str:
        """
        Constructs a string describing the current game state for the agent's prompt.
        """
        game_state = f"""
Current Game State:
Phase: {self.game.get_current_phase()} ({phase_type})
Year: {state['name']}

England's Status:
Centers: {sorted(list(self.power.centers))}
Units: {sorted(list(self.power.units))}
Supply Centers Needed: {self.power.influence}

Other Powers' Status:
{chr(10).join(f"{power}: {len(centers)} centers, {len(self.game.powers[power].units)} units" 
              for power, centers in self.game.get_centers().items() if power != 'ENGLAND')}
"""
        # Determine instructions based on phase type
        instructions = (
            "Submit one movement order for each unit." if phase_type == "M" else
            "Submit one retreat order for each dislodged unit." if phase_type == "R" else
            f"{'Build' if len(self.power.centers) > len(self.power.units) else 'Disband'} {abs(len(self.power.centers) - len(self.power.units))} units."
            if phase_type == "A" else "Unknown phase type"
        )
        game_state += f"\nInstructions:\n1. This is a {phase_type} phase. {instructions}\n2. Orders must be valid according to the current phase and game rules.\n"
        return game_state

    async def create_agent(self) -> AgentExecutor:
        """
        Creates the LangChain agent for England using ChatOpenAI and binds the submit_moves tool.
        """
        llm = ChatOpenAI(temperature=0.7)
        map_info = self.game.map.loc_abut
        
        system_prompt = f"""You are playing as England in a game of Diplomacy. Your goal is to expand your territory
and eventually dominate Europe. You should analyze the current game state and make strategic moves for all your units.

Key points to remember:
1. Orders must be valid according to Diplomacy rules.
2. Think strategically about supporting other units and coordinating movements.
3. Consider both offensive and defensive positions.
4. Pay attention to the current phase type:
   - MOVEMENT (M): Submit one order per unit.
   - RETREAT (R): Submit one order per dislodged unit.
   - ADJUSTMENT (A): Submit build or disband orders based on supply center count.

When submitting moves, use the standard Diplomacy notation.
Map Information (territory: [adjacent territories]):
{chr(10).join(f"{loc}: {neighbors}" for loc, neighbors in map_info.items())}

You MUST use the submit_moves tool to validate and submit your orders.
If the orders are invalid, analyze the error message and try again with corrected orders.
Submit your moves as a list of strings, e.g.: ["A LON H", "F NTH - NWY"].
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Bind the submit_moves tool for the agent
        submit_moves_tool = StructuredTool.from_function(
            func=self.submit_moves,
            name="submit_moves",
            description=self.submit_moves.__doc__
        )
        tools = [submit_moves_tool]
        llm_with_tools = llm.bind_tools(tools)
        
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"])
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return self.agent
    
    async def get_orders(self) -> List[str]:
        """
        Retrieves validated orders from the LangChain agent.
        The method builds the current game state prompt, then uses the agent to obtain orders.
        It retries up to a maximum number of attempts, logging token usage via get_openai_callback.
        """
        if not self.agent:
            await self.create_agent()
        
        # Ensure no previous valid moves linger
        if hasattr(self, 'last_valid_moves'):
            delattr(self, 'last_valid_moves')
        
        state = self.game.get_state()
        phase_type = self.game.phase_type
        game_state = self._build_game_state(phase_type, state)
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    game_state += f"\n\nThis is attempt {attempt + 1} of {max_attempts}. Previous attempts failed. Please ensure all orders are valid."
                
                with get_openai_callback() as cb:
                    result = await self.agent.ainvoke({
                        "input": game_state,
                        "chat_history": self.chat_history
                    })
                    print(f"\nTurn {self.game.get_current_phase()} Usage:")
                    print(f"  Prompt Tokens: {cb.prompt_tokens}")
                    print(f"  Completion Tokens: {cb.completion_tokens}")
                    print(f"  Total Tokens: {cb.total_tokens}")
                    print(f"  Total Cost: ${cb.total_cost:.4f}")
                
                if hasattr(self, 'last_valid_moves'):
                    self.chat_history.append(("human", game_state))
                    self.chat_history.append(("assistant", f"Orders submitted: {self.last_valid_moves}"))
                    return self.last_valid_moves
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_attempts - 1:
                    raise Exception("Failed to get valid orders after maximum attempts")
        raise Exception("Failed to get valid orders")

async def main():
    """
    Main function to initialize the game, retrieve orders for England using the agent,
    generate random moves for other powers, process the turn, and display updated game state.
    """
    # Initialize the game and the England agent
    game = Game()
    england_agent = EnglandAgent(game)
    
    # Retrieve England's orders from the agent
    try:
        england_orders = await england_agent.get_orders()
        game.set_orders('ENGLAND', england_orders)
    except Exception as e:
        print(f"Error getting England's orders: {str(e)}")
        # Fallback: hold or disband based on phase
        if game.phase_type == 'M':
            england_orders = [f"{unit} H" for unit in game.powers['ENGLAND'].units]
        else:
            england_orders = [f"{unit} D" for unit in game.powers['ENGLAND'].units]
        game.set_orders('ENGLAND', england_orders)
    
    # Generate random orders for the other powers
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
    
    # Process the turn and display the game state
    game.process()
    print(f"\nProcessed {game.phase} - {game.get_state()['name']}")
    for power_name, power in game.powers.items():
        print(f"{power_name}: {len(power.centers)} centers, {len(power.units)} units")
        if power.orders:
            print(f"  Orders: {power.orders}")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 