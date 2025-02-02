from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from typing import List
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Define the schema for moves
class MovesInput(BaseModel):
    moves: List[str]

# Define a tool for submitting moves
def submit_moves(moves: MovesInput) -> str:
    """Submit moves for the current turn. Each move should be a valid Diplomacy order."""
    # In a real implementation, this would interact with the game
    return f"Submitted moves: {moves.moves}"

# Create the tool
moves_tool = StructuredTool(
    name="submit_moves",
    description="Submit moves for the current turn. Moves should be in standard Diplomacy notation (e.g., 'F LON - NTH').",
    func=submit_moves,
    args_schema=MovesInput
)

# Create the model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

# Create the prompt
system_message = SystemMessagePromptTemplate.from_template("""You are playing as England in a game of Diplomacy. You are a skilled player focused on making strong tactical moves.
The game has just started - it's Spring 1901. You control:
- Fleet London (F LON)
- Fleet Edinburgh (F EDI)
- Army Liverpool (A LVP)

Make your opening moves using the submit_moves tool. Provide the moves in standard Diplomacy notation.
Consider strategic positions like Norway, the North Sea, and the English Channel.

Available tools: {tools}
Tool names: {tool_names}
""")

prompt = ChatPromptTemplate.from_messages([
    system_message,
    HumanMessagePromptTemplate.from_template("{input}"),
    ("ai", "{agent_scratchpad}"),
])

# Create the agent
agent = create_structured_chat_agent(model, [moves_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[moves_tool], verbose=True)

# Run the agent
if __name__ == "__main__":
    response = agent_executor.invoke({"input": "What are your moves for Spring 1901?"})
    print("\nFinal Response:", response["output"]) 