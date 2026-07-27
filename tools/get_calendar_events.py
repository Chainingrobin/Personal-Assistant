import ollama 

def get_calendar_events(days_ahead: int = 7) -> str:
    """
    Retrieves upcoming calendar events within a given time window.

    Args:
        days_ahead: How many days ahead to look for events.
    """
    print(f"\n[MOCK API] 📅 Fetching events for the next {days_ahead} days...")
    return (
        "Event 1: Project Demo - in 2 days at 2:00 PM\n"
        "Event 2: ML Lecture - tomorrow at 10:00 AM\n"
        "Event 3: Study Group - in 5 days at 6:00 PM"
    )


# 1. Initialize the conversation history
messages = [
    {
        'role': 'system', 
        'content': 'You are a strict routing assistant. If asked about calender events, you MUST use the get_calendar_events tool. Do not ask the user for missing arguments, just leave them empty.'
    },
    {'role': 'user', 'content': 'What events do I have coming up in the next week?'}
]

max_retries = 10
attempt = 1
tool_called = False

print("Starting Orchestrator Retry Loop...\n")

# 2. The Retry Loop
while attempt <= max_retries:
    print(f"--- Attempt {attempt} of {max_retries} ---")
    
    response = ollama.chat(
        model='qwen2.5:3b',
        messages=messages,
        tools=[get_calendar_events],
        options={'temperature': 0.0}
    )
    
    response_message = response.get('message', {})
    
    # Check if the model successfully called the tool
    if response_message.get('tool_calls'):
        print("\n✅ SUCCESS: The model decided to call a tool!")
        for call in response_message['tool_calls']:
            print(f"Tool Name: {call['function']['name']}")
            print(f"Arguments: {call['function']['arguments']}")
        
        tool_called = True
        break # Exit the loop, we got the JSON we need!
        
    else:
        # The model failed and replied with text
        text_reply = response_message.get('content', '')
        print(f"❌ FAIL: Model replied with text: '{text_reply}'")
        
        # Add the model's incorrect response to the history so it has context
        messages.append(response_message)
        
        # Add a stern correction as a new user prompt
        messages.append({
            'role': 'user',
            'content': 'SYSTEM CORRECTION: You failed to use the required tool. Do not speak to me. Call the read_email tool immediately.'
        })
        
        attempt += 1

# 3. Fallback if it completely fails
if not tool_called:
    print("\n🚨 CRITICAL ERROR: Model failed to call the tool after maximum retries.")