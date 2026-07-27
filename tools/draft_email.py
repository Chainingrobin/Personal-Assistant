import ollama 

def draft_email(recipient: str, subject: str, body: str) -> str:
    """
    Drafts an email ready to be sent via the Gmail API.

    Args:
        recipient: The email address of the recipient.
        subject: The subject line of the email.
        body: The full body text of the email.
    """
    print(f"\n[MOCK API] ✉️  Drafting email...")
    print(f"  To: {recipient}")
    print(f"  Subject: {subject}")
    print(f"  Body preview: {body[:80]}...")
    return f"[DRAFT READY] Email to {recipient} drafted. Call send_email() to dispatch."

# 1. Initialize the conversation history
messages = [
    {
        'role': 'system', 
        'content': 'You are a strict routing assistant. If asked about emails, you MUST use the draft_email tool. Do not ask the user for missing arguments, just leave them empty.'
    },
    {'role': 'user', 'content': 'Draft an email for me and send to alex@company.com with subject "Meeting Update" and body "The meeting has been changed "'}
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
        tools=[draft_email],
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