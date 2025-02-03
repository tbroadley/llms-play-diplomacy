from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
import diplomacy
import asyncio
from datetime import datetime
from player import POWERS, Player
from game import DiplomacyGame


async def play_game(turn_time_limit: int, max_turns: int):
    """Play the game for a specified number of turns."""
    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client
    client = AsyncOpenAI()

    game = DiplomacyGame(turn_time_limit)
    renderer = diplomacy.engine.renderer.Renderer(game.game)
    turn_count = 0

    output_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting new game with all powers...")
    print("-" * 50)

    players = {power_name: Player(game, power_name, client) for power_name in POWERS}

    while not game.is_game_over() and turn_count < max_turns:
        print(f"\nTurn {turn_count + 1} - {game.game.phase}")
        print("-" * 50)

        tasks = []
        for power_name, player in players.items():
            task = asyncio.create_task(player.run())
            tasks.append(task)

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=turn_time_limit,
            )
            print(f"Results: {results}")
        except asyncio.TimeoutError:
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete/cancel
            await asyncio.gather(*tasks, return_exceptions=True)

        # Process the turn
        game.process_turn()

        turn_count += 1
        print(f"\nTurn {turn_count} completed")

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
    asyncio.run(play_game(turn_time_limit=15, max_turns=100))
