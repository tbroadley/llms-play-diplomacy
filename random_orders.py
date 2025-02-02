import random
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format

def main():
    # Create a new game
    game = Game()
    
    # Get the list of possible orders for each power
    while not game.is_game_done:
        # Get all possible orders for all units
        possible_orders = game.get_all_possible_orders()
        
        for power_name, power in game.powers.items():
            if power.units:  # Only process powers that have units
                power_orders = []
                
                # For each unit, randomly select a valid order
                for unit in power.units:
                    location = unit.split()[1]  # Get the location part of the unit (e.g., 'BUD' from 'A BUD')
                    unit_possible_orders = possible_orders.get(location, [])
                    if unit_possible_orders:
                        power_orders.append(random.choice(unit_possible_orders))
                
                # Set the orders for the power
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
    main() 