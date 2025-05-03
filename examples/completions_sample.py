from openai import OpenAI
import os
import sys
from dead_simple_self_learning import SelfLearner

client = OpenAI(api_key="YOUR_API_KEY")
task = "Write a sample email for a customer complaining about a product"
base_prompt = "You are a helpful assistant that can answer questions and help with tasks."
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "system", "content": base_prompt},{"role": "user", "content": task}]   
)

print("###### WITHOUT FEEDBACK ######")
print(response.choices[0].message.content)  

learner = SelfLearner(
    embedding_model="miniLM",
    memory_path="memory_completions .json",
    similarity_threshold=0.85,
    max_matches=2,
    llm_feedback_selection_layer="openai",
)
feeback = "Keep it under 4 sentences and concise"
learner.save_feedback(task, feeback)

print("###### WITH FEEDBACK ######")

enhanced_prompt = learner.apply_feedback(task, feeback)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "system", "content": enhanced_prompt},{"role": "user", "content": task}]   
)

print(response.choices[0].message.content)  