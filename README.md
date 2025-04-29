# Dead Simple Self-Learning

A lightweight Python library that allows any LLM agent to self-improve through feedback, without retraining models.

## 📋 Overview

This library provides a simple system for capturing, storing, and reusing feedback for LLM tasks. It works by:

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
- **Customizable**: Configurable thresholds, formatters, and memory handling
- **Zero Infrastructure**: Works out of the box with minimal setup

## 🔧 Installation

Once published, you'll be able to install the package via pip:

```bash
pip install dead_simple_self_learning
```

Until then, you can install from source:

```bash
git clone https://github.com/yourusername/dead_simple_self_learning.git
cd dead_simple_self_learning
pip install -e .
```

## 🚀 Quick Start

```python
import os
from openai import OpenAI
from dead_simple_self_learning import SelfLearner

# Set your OpenAI API key (if using OpenAI)
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize OpenAI client
client = OpenAI()

# Initialize a self-learner
learner = SelfLearner(embedding_model="miniLM")  # No API key needed for miniLM

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

- `"openai"`: Uses GPT-4o to select the best feedback (requires API key)

## 🔄 How It Works

1. **Task Embedding**: When a new task comes in, it's embedded using the chosen model
2. **Similarity Search**: The system searches for similar tasks in memory
3. **Feedback Retrieval**: If similar tasks are found, their feedback is retrieved
4. **Selection Process**: If multiple similar tasks are found, OpenAI's GPT-4o is used to select the best feedback
5. **Prompt Enhancement**: The selected feedback is injected into the base prompt
6. **Usage Tracking**: The system tracks which feedback is most useful

## 📚 Advanced Usage

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

## 💻 Requirements

- Python 3.7+
- numpy >=1.20.0
- sentence-transformers >=2.2.0 (for local embedding models)
- openai >=1.0.0 (optional, for OpenAI embeddings and LLM)

## 📦 Publishing Your Package

This repository includes a `setup.py` file, which is the configuration file for publishing your package to PyPI (Python Package Index). Here's how to use it:

1. **Update package information**: Review and update the metadata in setup.py if needed
2. **Build the package**:
   ```bash
   pip install build
   python -m build
   ```
3. **Test locally**:
   ```bash
   pip install dist/dead_simple_self_learning-0.1.0-py3-none-any.whl
   ```
4. **Publish to PyPI**:
   ```bash
   pip install twine
   twine upload dist/*
   ```

You'll need a PyPI account, and you'll be prompted for your username and password during upload.

## 🔍 Examples

Check out the `examples/` directory for more detailed examples:

- `demo_sample_example.py`: Quick start example showing the basic workflow
- `feedback_operations_example.py`: Demonstrates various feedback operations

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the library.

## 📜 License

MIT 