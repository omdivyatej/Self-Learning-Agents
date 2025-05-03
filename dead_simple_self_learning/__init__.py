"""
dead_simple_self_learning: A lightweight library for LLM self-improvement through feedback

This library allows any LLM agent to self-improve through feedback without retraining.
It provides a simple, transparent way to apply historical feedback to new tasks.
Supports both synchronous and asynchronous operations for better performance in I/O-bound contexts.

Key components:
- SelfLearner: Core class that manages feedback application
- Memory: Handles persistent storage of feedback
- Embedder: Generates embeddings for semantic similarity matching

Version: 0.1.0
"""

import logging

# Configure root package logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a null handler to avoid "No handler found" warnings
# Users can configure their own handlers as needed
null_handler = logging.NullHandler()
logger.addHandler(null_handler)

# Export public classes
from .embedder import Embedder
from .memory import Memory
from .learner import SelfLearner

__version__ = "1.1.1" 