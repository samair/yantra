"""Interactive demonstration of Yantra Agent and Tool Dispatching."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yantra import Agent, tool


# 1. Define tools using @tool decorator
@tool
def set_thermostat(temperature: int, mode: str = "auto") -> str:
    """Set the HVAC thermostat temperature in degrees and operating mode."""
    return f"[SUCCESS] Thermostat set to {temperature}F (mode={mode})"


@tool
def control_lights(room: str, state: str = "on", brightness: int = 100) -> str:
    """Control smart lighting in a specified room."""
    return f"[SUCCESS] {room.capitalize()} lights turned {state.upper()} at {brightness}% brightness"


@tool
def get_weather(city: str) -> str:
    """Fetch current real-time weather conditions for a city."""
    return f"[SUCCESS] Current weather in {city}: 72F, Sunny, Wind 5mph"


@tool
def schedule_meeting(title: str, time: str) -> str:
    """Schedule a meeting on the calendar."""
    return f"[SUCCESS] Scheduled '{title}' for {time}"


def main():
    print("=" * 65)
    print("      YANTRA: Sub-50MB On-Device Tool Calling Agent Demo")
    print("=" * 65)

    tools = [set_thermostat, control_lights, get_weather, schedule_meeting]
    print(f"Registered Tools: {[t.name for t in tools]}\n")

    # Initialize agent (uses default model or INT8 quantized checkpoint if available)
    checkpoint_path = Path("checkpoints/yantra_35m_int8.bin")
    if checkpoint_path.exists():
        print(f"Loading INT8 quantized model from {checkpoint_path}...")
        agent = Agent(tools=tools, model=str(checkpoint_path))
    else:
        print("Instantiating base Yantra model...")
        agent = Agent(tools=tools)

    # Test cases
    queries = [
        "Turn on the kitchen lights and dim to 50 percent",
        "What's the weather like in Tokyo right now?",
        "Please schedule Team Sync at 3pm",
        "Set thermostat to 72 degrees",
        "Can you write a poem about autumn leaves?",  # Unsupported / refusal case
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}] User Query: \"{q}\"")
        # Run agent
        response = agent.run(q, execute_tools=True, max_new_tokens=64)

        if response.is_refusal:
            print("  Result: REFUSAL / NO TOOL DISPATCHED")
            print(f"  Confidence: {response.confidence:.2f}")
        else:
            print(f"  Dispatched Calls: {response.tool_calls}")
            print(f"  Executed Outputs: {response.results}")
            print(f"  Confidence: {response.confidence:.2f}")

    print("\n" + "=" * 65)
    print("Demo execution completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
