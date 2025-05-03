# Dead Simple Self-Learning

A lightweight Python library that allows any LLM agent to self-improve through feedback, without retraining models.

<p align="center">
  <img src="https://cdn.iconscout.com/icon/premium/png-256-thumb/multi-agent-2134465-1802462.png?f=webp&w=256" width="200" alt="Dead Simple Self-Learning Logo">
</p>

## 📋 Overview

**Problem**: LLM agents struggle to consistently learn from user feedback without requiring costly model retraining or complex infrastructure.

**Solution**: This library provides a simple system for capturing, storing, and reusing feedback for LLM tasks. It works by:

1. Collecting feedback on LLM outputs
2. Storing this feedback with embeddings of the original task
3. Retrieving relevant feedback for similar future tasks (feedback selection layer: only openai right now)
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
print("#######################Original output:", original)

# Save feedback for the task
feedback = "Keep it under 100 words and focus on benefits not features"
learner.save_feedback(task, feedback)

# Apply feedback to the prompt
enhanced_prompt = learner.apply_feedback(task, base_prompt)
enhanced = generate_text(enhanced_prompt, task)

print("######################Improved output:", enhanced)
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