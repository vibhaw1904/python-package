#!/usr/bin/env python3
"""Example usage of Nugen Chat Model."""

import os
from dotenv import load_dotenv
from langchain_nugen.chat_models.nugen import ChatNugen
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

def basic_chat_example():
    """Basic chat example with Nugen."""
    # Initialize the chat model
    chat = ChatNugen(
        api_key=os.getenv("NUGEN_API_KEY"),
        model_name="nugen-flash-instruct",
        temperature=0.7,
        max_tokens=500
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello! Can you tell me about artificial intelligence?")
    ]
    
    # Get response
    try:
        response = chat.invoke(messages)
        print("Response:", response.content)
    except Exception as e:
        print(f"Error: {e}")

def streaming_example():
    """Example of streaming responses."""
    chat = ChatNugen(
        api_key=os.getenv("NUGEN_API_KEY"),
        model_name="nugen-flash-instruct"
    )
    
    messages = [HumanMessage(content="Write a short story about a robot.")]
    
    try:
        print("Streaming response:")
        for chunk in chat.stream(messages):
            print(chunk.content, end="", flush=True)
        print()  # New line after streaming
    except Exception as e:
        print(f"Error: {e}")

def environment_variable_example():
    """Example using environment variable for API key."""
    # API key is automatically loaded from .env file
    
    chat = ChatNugen(
        model_name="nugen-flash-instruct",
        temperature=0.5
    )
    
    messages = [HumanMessage(content="What is the capital of France?")]
    
    try:
        response = chat.invoke(messages)
        print("Response:", response.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Nugen Chat Model Examples")
    print("=" * 30)
    
    print("\n1. Basic Chat Example:")
    basic_chat_example()
    
    print("\n2. Streaming Example:")
    streaming_example()
    
    print("\n3. Environment Variable Example:")
    environment_variable_example()