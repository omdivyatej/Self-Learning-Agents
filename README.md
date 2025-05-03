# Dead Simple Self-Learning

A lightweight Python library that allows any LLM agent to self-improve through feedback, without retraining models.

## 📋 Overview

**Problem**: LLM agents struggle to consistently learn from user feedback without requiring costly model retraining or complex infrastructure.

**Solution**: This library provides a simple system for capturing, storing, and reusing feedback for LLM tasks. It works by:

1. Collecting feedback on LLM outputs
2. Storing this feedback with embeddings of the original task
3. Retrieving relevant feedback for similar future tasks
4. Enhancing prompts with the feedback to improve results

All of this happens without any model retraining - just by enhancing prompts with contextual feedback.

## ✨ Features

- **Simple API**: Just a few methods to enhance prompts and save feedback
- **Multiple Embedding Models**: Support for OpenAI and HuggingFace models (MiniLM, BGE-small)
- **Local-First**: Uses JSON files for storage with no external DB requirements
- **Smart Feedback Selection**: Uses OpenAI to choose the most relevant feedback for a task
- **Async Support**: Both synchronous and asynchronous APIs for better performance
- **Customizable**: Configurable thresholds, formatters, and memory handling
- **Zero Infrastructure**: Works out of the box with minimal setup
- **Framework Agnostic**: Works with any LLM provider (OpenAI, Anthropic, etc.)
- **Integration Examples**: Ready-to-use examples with LangChain, Agno, and more

## 🔧 Installation

You can install the package via pip:

```bash
pip install dead_simple_self_learning
```

### Dependencies

- **Required**: 
  - Python 3.7+
  - numpy >=1.20.0
  - sentence-transformers >=2.2.0

- **Optional**:
  - openai >=1.0.0 (for OpenAI embeddings and LLM feedback selection)
  - langchain, agno (for specific integration examples)

Install with optional OpenAI dependency:
```bash
pip install "dead_simple_self_learning[openai]"
```

Install for development:
```bash
pip install "dead_simple_self_learning[dev]"
```

## 🚀 Quick Start

```python
from openai import OpenAI
from dead_simple_self_learning import SelfLearner

# Initialize OpenAI client (you need your own API key)
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Initialize a self-learner (no API key needed for miniLM)
learner = SelfLearner(embedding_model="miniLM")

# Define our task and original prompt
task = "Write a product description for a smartphone"
base_prompt = "You are a copywriter."

# Generate text without feedback
def generate_text(prompt, task):
    return client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": task}]
    ).choices[0].message.content

# Generate original text
original = generate_text(base_prompt, task)
print("Original output:", original)

# Save feedback for the task
feedback = "Keep it under 100 words and focus on benefits not features"
learner.save_feedback(task, feedback)

# Apply feedback to the prompt
enhanced_prompt = learner.apply_feedback(task, base_prompt)
enhanced = generate_text(enhanced_prompt, task)

print("Improved output:", enhanced)
```

## 📊 Package Structure

```
dead_simple_self_learning/
├── __init__.py         # Package exports
├── __main__.py         # CLI entrypoint
├── embedder.py         # Handles embedding generation
├── memory.py           # Manages storage and retrieval
└── learner.py          # Core functionality
```

## 📖 Detailed Guide

### Core Components

#### Embedder

The Embedder class generates vector embeddings for tasks:

```python
from dead_simple_self_learning import Embedder

# Use a HuggingFace model (no API key required)
embedder = Embedder(model_name="miniLM")  

# Use OpenAI (requires API key in env var OPENAI_API_KEY)
embedder = Embedder(model_name="openai")  

# Generate an embedding
vector = embedder.embed("your text here")
```

#### Memory

The Memory class handles storage and retrieval of feedback:

```python
from dead_simple_self_learning import Memory

memory = Memory(file_path="my_memory.json")

# Add a feedback entry
memory.add_entry(
    task="Task description",
    feedback="The feedback to remember",
    embedding=[0.1, 0.2, 0.3, ...]  # Vector from Embedder
)

# Find similar tasks
similar = memory.find_similar(
    embedding=[0.1, 0.2, 0.3, ...],
    threshold=0.85,  # Similarity threshold
    top_k=2  # Number of results to return
)

# Other operations
memory.reset()  # Clear all memories
all_entries = memory.get_all()  # Get all feedback entries
```

#### SelfLearner

The main class that brings everything together:

```python
from dead_simple_self_learning import SelfLearner

learner = SelfLearner(
    embedding_model="miniLM",               # Embedding model to use
    memory_path="memory.json",              # Where to store feedback
    similarity_threshold=0.85,              # Minimum similarity for matches
    max_matches=2,                          # Max number of matches to consider
    llm_feedback_selection_layer="openai"   # LLM for feedback selection
)

# Core functionality
enhanced_prompt = learner.apply_feedback("Task description", "Base prompt")
learner.save_feedback("Task description", "The feedback to save")

# Async variants (for better performance in async contexts)
enhanced_prompt = await learner.apply_feedback_async("Task", "Prompt")
await learner.save_feedback_async("Task", "Feedback")

# Configuration
learner.set_similarity_threshold(0.75)
learner.set_max_matches(3)

# Custom feedback formatting
def my_formatter(base_prompt, feedback):
    return f"{base_prompt}\n\n[IMPORTANT]: {feedback}"
    
learner.set_feedback_formatter(my_formatter)

# Memory management
learner.export_memory("backup.json")
learner.import_memory("external_memory.json")
learner.reset_memory()
```

### Configuration Options

#### Embedding Models

- `"openai"`: Uses OpenAI's `text-embedding-ada-002` (requires API key)
- `"miniLM"`: Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`
- `"bge-small"`: Uses HuggingFace's `BAAI/bge-small-en`

#### LLM Feedback Selection

- `"openai"`: Uses GPT models to select the best feedback (requires API key)

## 🔄 How It Works

1. **Task Embedding**: When a new task comes in, it's embedded using the chosen model
2. **Similarity Search**: The system searches for similar tasks in memory
3. **Feedback Retrieval**: If similar tasks are found, their feedback is retrieved
4. **Selection Process**: If multiple similar tasks are found, the best feedback is selected
5. **Prompt Enhancement**: The selected feedback is injected into the base prompt
6. **Usage Tracking**: The system tracks which feedback is most useful

## 📚 Advanced Usage

### Asynchronous API

For I/O-bound applications, use the async methods for better performance:

```python
import asyncio
from dead_simple_self_learning import SelfLearner

async def enhance_prompts():
    learner = SelfLearner(embedding_model="miniLM")
    
    # Save feedback asynchronously
    await learner.save_feedback_async(
        "Write a product description", 
        "Focus on unique features"
    )
    
    # Apply feedback asynchronously
    enhanced = await learner.apply_feedback_async(
        "Write a product description for headphones",
        "You are a copywriter."
    )
    
    return enhanced

# Run the async function
enhanced_prompt = asyncio.run(enhance_prompts())
```

### Feedback Operations

The library provides several operations for managing feedback:

```python
# List all stored feedback
learner.list_all_feedback(verbose=True)

# List feedback for a specific task
learner.list_feedback(task="Write a product description")

# Find feedback containing substring
learner.list_feedback_substring(task_substring="email")

# Remove feedback by index
learner.remove_feedback(index=1)

# Remove feedback for specific task
learner.remove_feedback_for_task(task="Write a product description")

# Export/import memory
learner.export_memory("backup.json")
learner.import_memory("external_memory.json")
```

### Custom Feedback Formatting

You can customize how feedback is injected into the base prompt:

```python
def custom_formatter(base_prompt, feedback):
    return f"""
{base_prompt}

IMPORTANT GUIDANCE:
- {feedback}
- Always aim for clarity and simplicity
"""

learner = SelfLearner(
    embedding_model="miniLM",
    feedback_formatter=custom_formatter
)
```

## 🔍 Examples

Check out the `examples/` directory for more detailed examples:

- `demo_sample_example.py`: Quick start example showing the basic workflow
- `feedback_operations_example.py`: Demonstrates various feedback operations
- `lanchain_sql_agent_example.py`: Integration with LangChain SQL agent
- `agno_sample_demo.py`: Integration with Agno framework

To run examples:

```bash
# Install the package first (from PyPI or in development mode)
pip install -e .

# Run an example
python examples/demo_sample_example.py
```

> **⚠️ Important:** The examples contain placeholders for API keys. Replace `YOUR_API_KEY_HERE` with your actual API key before running examples that require OpenAI or other API access.

## ⚠️ Security Warning

- Never hardcode API keys in your code
- Use environment variables or secure vaults to store sensitive credentials
- The examples in this repository use placeholder API keys
- Set your API key using:
  ```python
  import os
  os.environ["OPENAI_API_KEY"] = "your-key-here"  # Not recommended for production
  ```

## 🔍 Troubleshooting

### Common Issues

1. **Missing dependencies**: If using OpenAI features, install the OpenAI package
   ```bash
   pip install "dead_simple_self_learning[openai]"
   ```

2. **Embedding model loading errors**: Ensure you have enough disk space and RAM for the embedding models

3. **Performance issues**: For high-throughput applications, use the async API

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Run tests: `pytest`
6. Submit a pull request

## 📦 Publishing

This repository includes a `setup.py` file for PyPI publishing:

1. Update version in `dead_simple_self_learning/__init__.py`
2. Build the package:
   ```bash
   pip install build
   python -m build
   ```
3. Test locally:
   ```bash
   pip install dist/dead_simple_self_learning-0.1.0-py3-none-any.whl
   ```
4. Publish to PyPI:
   ```bash
   pip install twine
   twine upload dist/*
   ```

## 📜 License

MIT 