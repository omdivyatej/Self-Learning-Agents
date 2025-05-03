import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from dead_simple_self_learning import SelfLearner

base_prompt = "You are an enthusiastic news reporter with a flair for storytelling!"
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description=base_prompt,
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)
task = "Write a story about a breaking news story from New York"
print("######WITHOUT FEEDBACK####")
result = agent.run(task, stream=False)
print(result.content)

learner = SelfLearner(
    embedding_model="miniLM",
    memory_path="memory.json",
    similarity_threshold=0.85,
    max_matches=2,
    llm_feedback_selection_layer="openai",
)

feeback = "keep it under 4 sentences and concise"
learner.save_feedback(task, feeback)
enhanced_prompt = learner.apply_feedback(task, base_prompt)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description=enhanced_prompt,
    markdown=True
)

print("###### WITH FEEDBACK ######")
result_with_feedback = agent.run(task, stream=False)
print(result_with_feedback.content)