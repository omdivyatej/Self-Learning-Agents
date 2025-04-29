import os
import sys
import asyncio
from openai import OpenAI
from openai import AsyncOpenAI

try:
    from dead_simple_self_learning import SelfLearner
except ImportError:
    # Fallback for running the example without installing the package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dead_simple_self_learning import SelfLearner

client = OpenAI()
async_client = AsyncOpenAI()

def generate_text(prompt, task):
    return client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": task}]
    ).choices[0].message.content

async def generate_text_async(prompt, task):
    response = await async_client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": task}]
    )
    return response.choices[0].message.content

def run_sync_example():
    print("\n=== Running Synchronous Example ===")
    
    # Initialize the self-learner with default settings
    learner = SelfLearner(
        embedding_model="miniLM",   # Use miniLM for embeddings (no API key needed)
        clear_memory=True,          # Start with fresh memory
    )
    
    # Define our task and base prompt
    task = "Write a product description for a smartphone"
    original_prompt = "You are a copywriter."
    
    # Generate text without feedback
    print("\nGenerating text WITHOUT feedback...")
    original = generate_text(original_prompt, task)
    
    print("\nOriginal output:")
    print(original)
    
    # Save feedback for the task
    feedback = "Keep it under 100 words and focus on benefits not features"
    print(f"\nSaving feedback: '{feedback}'")
    learner.save_feedback(task, feedback)
    
    # Apply feedback to the prompt
    print("\nGenerating text WITH feedback...")
    enhanced_prompt = learner.apply_feedback(task, original_prompt)
    enhanced = generate_text(enhanced_prompt, task)
    
    print("\nImproved output:")
    print(enhanced)
    
    print("\nDone! Synchronous example completed.")

async def run_async_example():
    print("\n=== Running Asynchronous Example ===")
    
    # Initialize the self-learner with default settings
    learner = SelfLearner(
        embedding_model="miniLM",   # Use miniLM for embeddings (no API key needed)
        memory_path="async_memory.json",  # Use a different memory file
        clear_memory=True,          # Start with fresh memory
    )
    
    # Define our task and base prompt
    task = "Write a marketing email for a fitness app"
    original_prompt = "You are a marketing specialist."
    
    # Generate text without feedback
    print("\nGenerating text WITHOUT feedback...")
    original = await generate_text_async(original_prompt, task)
    
    print("\nOriginal output:")
    print(original)
    
    # Save feedback for the task asynchronously
    feedback = "Be conversational and include a clear call-to-action. Keep it short."
    print(f"\nSaving feedback: '{feedback}'")
    await learner.save_feedback_async(task, feedback)
    
    # Apply feedback to the prompt asynchronously
    print("\nGenerating text WITH feedback...")
    enhanced_prompt = await learner.apply_feedback_async(task, original_prompt)
    enhanced = await generate_text_async(enhanced_prompt, task)
    
    print("\nImproved output:")
    print(enhanced)
    
    print("\nDone! Asynchronous example completed.")

def main():
    # Run the synchronous example
    run_sync_example()
    
    # Run the async example using asyncio
    asyncio.run(run_async_example())
    
    print("\nBoth examples completed. See the other examples for feedback operations and advanced features.")

if __name__ == "__main__":
    main() 