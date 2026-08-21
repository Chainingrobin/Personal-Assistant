import ollama
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import retrieve, format_context

# Add user_id as a keyword-only argument to match the other tools
def query_rag(query: str, top_k: int = 3, *, user_id: str = "") -> str:
    """
    Queries the local RAG knowledge base for relevant user context.
    Do NOT use this for general world knowledge or trivia.

    Args:
        query: The natural language query to search the knowledge base.
        top_k: The number of top results to return.
    """
    print(f"\n[RAG] 🔍 Querying knowledge base: '{query}'")
    # Eventually, you can pass user_id into retrieve() to isolate documents!
    results = retrieve(query, top_k=top_k, user_id=user_id)
    return format_context(results)

if __name__ == "__main__":
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
                'content': 'SYSTEM CORRECTION: You failed to use the required tool. Do not speak to me. Call the query_rag tool immediately.'
            })
            
            attempt += 1

    # 3. Fallback if it completely fails
    if not tool_called:
        print("\n🚨 CRITICAL ERROR: Model failed to call the tool after maximum retries.")