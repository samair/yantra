"""Interactive CLI Shell to chat and test tools with Yantra in real-time."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yantra import Agent, tool


# Define sample tools
@tool
def set_thermostat(temperature: int, mode: str = "auto") -> str:
    """Adjust HVAC thermostat temperature in degrees and operating mode."""
    return f"[THERMOSTAT] Temperature set to {temperature}F (mode={mode})"


@tool
def control_light(room: str, state: str = "on", brightness: int = 100) -> str:
    """Turn on/off or dim lighting fixtures in a designated room."""
    return f"[LIGHTS] {room.capitalize()} lights turned {state.upper()} at {brightness}% brightness"


@tool
def get_weather(city: str) -> str:
    """Fetch current real-time weather conditions for a city."""
    return f"[WEATHER] Current weather in {city}: 72F, Clear skies"


@tool
def create_calendar_event(title: str, time: str) -> str:
    """Create an event on the user's primary calendar."""
    return f"[CALENDAR] Scheduled '{title}' for {time}"


def main():
    print("=" * 60)
    print("  YANTRA: Interactive Tool Calling Shell (< 50MB Model)")
    print("=" * 60)

    tools = [set_thermostat, control_light, get_weather, create_calendar_event]
    print("Available Tools:")
    for t in tools:
        print(f"  • {t.name}: {t.description}")

    checkpoint_path = Path("checkpoints/yantra_int8.bin")
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/yantra_latest.pt")

    print(f"\nLoading model from {checkpoint_path}...")
    agent = Agent(tools=tools, model=str(checkpoint_path))
    print("Ready! Type a command or 'exit' / 'quit' to leave.\n")

    while True:
        try:
            query = input("User > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            response = agent.run(query, execute_tools=True, max_new_tokens=128)

            if response.thought:
                print(f"  [Thought]: {response.thought}")

            if response.is_refusal:
                print(f"  [Output] : Refusal / No matching tool found (Confidence: {response.confidence:.2f})")
            else:
                print(f"  [Calls]  : {response.tool_calls}")
                print(f"  [Result] : {response.results}")

            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
