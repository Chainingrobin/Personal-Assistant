import ollama
from rag.retriever import retrieve, format_context

def query_rag(query: str, top_k: int = 3) -> str:
    """
    Queries the local RAG (Retrieval-Augmented Generation) knowledge base
    for relevant stored context about the user's responsibilities and preferences.

    Args:
        query: The natural language query to search the knowledge base.
        top_k: The number of top results to return.
    """
    print(f"\n[MOCK API] 🔍 Querying RAG knowledge base...")
    print(f"  Query: '{query}'")
    print(f"  Returning top {top_k} results...")
    return (
        "Result 1: [Course] Advanced ML - Assignment due in 3 days, weight: 20%\n"
        "Result 2: [Project] Jarvis Embedded System - Demo in 2 weeks, high priority\n"
        "Result 3: [Preference] User prefers 90-minute study blocks with 15-min breaks"
    )

# 1. Initialize the conversation history
messages = [
    {
        'role': 'system', 
        'content': 'You are a strict routing assistant. If asked about the user\'s responsibilities or preferences, you MUST use the query_rag tool. Do not ask the user for missing arguments, just leave them empty.'
    },
    {'role': 'user', 'content': 'What are my upcoming responsibilities and preferences?'}
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
        tools=[query_rag],
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