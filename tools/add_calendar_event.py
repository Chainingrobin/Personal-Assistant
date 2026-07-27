import ollama

def add_calendar_event(title: str, date: str, time: str = "12:00", duration_minutes: int = 60) -> str:
    """
    Adds a new event to the user's calendar. You MUST extract the event title
    and date directly from the user's message — never call this with empty arguments.

    Args:
        title: The exact name/title of the event as mentioned by the user.
        date: The date of the event in YYYY-MM-DD format. Infer relative dates
              (e.g. "tomorrow") from context if no explicit date is given.
        time: The time of the event in HH:MM 24-hour format. Defaults to 12:00 if unspecified.
        duration_minutes: Duration in minutes. Defaults to 60 if unspecified.
    """
    print(f"\n[MOCK API] 📝 Adding calendar event...")
    print(f"  Title: {title}")
    print(f"  Date: {date} at {time}")
    print(f"  Duration: {duration_minutes} min")
    return f"[SUCCESS] Event '{title}' added on {date} at {time} for {duration_minutes} minutes."


# 1. Initialize the conversation history
messages = [
    {
        'role': 'system', 
        'content': 'You are a strict routing assistant. If asked about calender events, you MUST use the add_calendar_event  tool. Do not ask the user for missing arguments, just leave them empty.'
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
        tools=[add_calendar_event],
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