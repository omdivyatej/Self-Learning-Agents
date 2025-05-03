import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dead_simple_self_learning import SelfLearner

# ====== Connect to the SQLite Database ======
db = SQLDatabase.from_uri("sqlite:///startup_metrics.db")

# ====== Create a language model ======
llm = ChatOpenAI(temperature=0.4, model="gpt-4o")

# ====== Create SQL Toolkit ======
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# ====== Setup SelfLearner Memory ======
MEMORY_PATH = "sql_agent_selflearner_memory.json"

learner = SelfLearner(
    embedding_model="miniLM",  # or "openai"
    memory_path=MEMORY_PATH,
    show_feedback_selection=True,
    temporary_memory=False,
    similarity_threshold=0.85,
    max_matches=2,
    llm_feedback_selection_layer="openai",
    feedback_formatter=None,
    clear_memory=False,
)

# ====== Define Base Prompt ======
BASE_SYSTEM_PROMPT = """You are an expert data analyst specialized in writing perfect SQL queries. 
You MUST use only the 'startup_metrics' table unless specified.
When in doubt, first ask the InfoSQLDatabaseTool for table info.

"""

# ====== Build SQL Agent Dynamically ======
def build_sql_agent(task: str):
    # 1. Use SelfLearner to improve system prompt based on past feedback
    enhanced_system_prompt = learner.apply_feedback(task, BASE_SYSTEM_PROMPT)
    
    print(f"Enhanced System Prompt: {enhanced_system_prompt}") 
    # 2. Create a custom prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_system_prompt),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}")
    ])

    # 3. Create agent with enhanced prompt
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        top_k=10,
        prompt=prompt,  # Use the full custom prompt
        agent_type=AgentType.OPENAI_FUNCTIONS
    )

    return agent_executor

# ====== Function to run agent ======
def run_agent(task):
    agent_executor = build_sql_agent(task)
    result = agent_executor.invoke({"input": task})
    
    return result

# ====== Interactive mode ======
if __name__ == "__main__":
    print("SQL Agent initialized with Self-Learning Memory 🧠. Type 'exit' to quit.")
    print("You can ask questions about your startup metrics database.")
    print("Example: 'How many users are on the pro plan?'")

    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            print("Exiting SQL Agent. Goodbye!")
            break
        
        # 1. Run agent with memory-enhanced prompt
        result = run_agent(user_input)

        print("\nAgent Response:")
        print(result["output"])

        # 2. Optionally allow feedback
        give_feedback = input("\nDo you want to give feedback on this answer? (y/n): ")
        if give_feedback.lower() == 'y':
            feedback_text = input("Enter your feedback: ")
            learner.save_feedback(user_input, feedback_text)
            print("✅ Feedback saved! Agent will now learn from this.")