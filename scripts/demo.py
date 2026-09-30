"""Interactive demonstration of Yantra Agent and Tool Dispatching."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yantra import Agent, tool


# 1. Define tools using @tool decorator
@tool
def set_thermostat(temperature: int, mode: str = "auto") -> str:
    """Adjust HVAC thermostat temperature in degrees and mode."""
    return f"[SUCCESS] Thermostat set to {temperature}F (mode={mode})"


@tool
def control_light(room: str, state: str = "on", brightness: int = 100) -> str:
    """Turn on/off or dim lighting fixtures in a designated room."""
    return f"[SUCCESS] {room.capitalize()} lights set to {state.upper()} at {brightness}% brightness"


@tool
def get_weather(city: str) -> str:
    """Fetch current real-time weather conditions for a city."""
    return f"[SUCCESS] Current weather in {city}: 72F, Sunny, Wind 5mph"


@tool
def control_door(location: str, state:bool) -> str:
    """Close or Open door at a certain location."""
    return f"[SUCCESS] Perfomed  {state} action on door at {location}"


@tool
def create_calendar_event(title: str, time: str) -> str:
    """Create an event on the user's primary calendar."""
    return f"[SUCCESS] Scheduled '{title}' for {time}"


def main():
    print("=" * 65)
    print("      YANTRA: Sub-50MB On-Device Tool Calling Agent Demo")
    print("=" * 65)

    tools = [set_thermostat, control_light, get_weather, create_calendar_event]
    print(f"Registered Tools: {[t.name for t in tools]}\n")

    # Initialize agent (check for trained INT8 or base checkpoints)
    int8_checkpoint = Path("checkpoints/yantra_int8.bin")
    pt_checkpoint = Path("checkpoints/yantra_latest.pt")
    old_checkpoint = Path("checkpoints/yantra_35m_int8.bin")

    if int8_checkpoint.exists():
        print(f"Loading trained INT8 quantized model from {int8_checkpoint}...")
        agent = Agent(tools=tools, model=str(int8_checkpoint))
    elif pt_checkpoint.exists():
        print(f"Loading checkpoint from {pt_checkpoint}...")
        from yantra.model.transformer import YantraForToolCalling
        from yantra.model.config import YantraConfig
        import torch
        ckpt = torch.load(str(pt_checkpoint), map_location="cpu")
        m = YantraForToolCalling(YantraConfig(**ckpt["config"]))
        m.load_state_dict(ckpt["model_state_dict"])
        agent = Agent(tools=tools, model=m)
    elif old_checkpoint.exists():
        print(f"Loading model from {old_checkpoint}...")
        agent = Agent(tools=tools, model=str(old_checkpoint))
    else:
        print("Instantiating base Yantra model...")
        agent = Agent(tools=tools)

    # Test cases
    queries = [
        "Turn on the kitchen lights",
        "set the thermostat to 72 degrees",
        "schedule Team Sync at 3pm",
        "Close front door",
        "Can you write a poem about autumn leaves?",  # Unsupported / refusal case
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}] User Query: \"{q}\"")
        # Run agent
        response = agent.run(q, execute_tools=True, max_new_tokens=128)

        if response.thought:
            print(f"  Reasoning: {response.thought}")

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
