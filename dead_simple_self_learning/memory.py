"""
Memory module for storing and retrieving feedback data.

Handles local JSON storage of tasks, feedback, and embeddings.
"""

import os
import json
import logging
import numpy as np
import shutil
import tempfile
from typing import List, Dict, Any, Optional, Tuple

# Configure module logger
logger = logging.getLogger(__name__)

class Memory:
    """
    Memory class for storing and retrieving task feedback.
    Uses a local JSON file for persistent storage.
    """

    def __init__(self, file_path: str = "memory.json", temporary: bool = False):
        """
        Initialize the Memory with a specified storage file.

        Args:
            file_path (str): Path to the JSON file for storing memory.
                             Defaults to "memory.json" in the current directory.
                             Can specify a subdirectory like "project1/memory.json"
            temporary (bool): If True, the memory will be stored in a temporary file
                              that will be deleted when the program exits.
                              Defaults to False.
        """
        logger.debug(f"Initializing Memory with file_path={file_path}, temporary={temporary}")
        
        # Handle temporary memory option
        if temporary:
            self.temp_dir = tempfile.mkdtemp()
            self.file_path = os.path.join(self.temp_dir, os.path.basename(file_path))
            self.is_temporary = True
            logger.info(f"Using temporary memory file: {self.file_path}")
        else:
            self.file_path = file_path
            self.is_temporary = False
            logger.info(f"Using persistent memory file: {self.file_path}")
        
        self._ensure_memory_file()

    def _ensure_memory_file(self) -> None:
        """
        Ensure the memory file exists, create it if it doesn't.
        
        Creates parent directories if they don't exist.
        Initializes an empty memory file if it doesn't exist.
        """
        try:
            if not os.path.exists(self.file_path):
                # Create directory if needed
                directory = os.path.dirname(self.file_path)
                if directory and not os.path.exists(directory):
                    logger.debug(f"Creating directory: {directory}")
                    os.makedirs(directory)
                
                # Create empty memory file
                logger.debug(f"Initializing empty memory file: {self.file_path}")
                self._save_memory([])
            else:
                logger.debug(f"Memory file already exists: {self.file_path}")
        except Exception as e:
            logger.error(f"Error ensuring memory file exists: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to create memory file: {str(e)}")

    def _load_memory(self) -> List[Dict[str, Any]]:
        """
        Load memory data from the JSON file.

        Returns:
            List[Dict[str, Any]]: List of memory entries.
            
        Raises:
            RuntimeError: If memory file cannot be loaded.
        """
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                logger.debug(f"Loaded {len(data)} entries from memory file")
                return data
        except json.JSONDecodeError as e:
            logger.warning(f"Memory file contains invalid JSON: {str(e)}")
            logger.info("Returning empty memory")
            return []
        except FileNotFoundError:
            logger.warning(f"Memory file not found: {self.file_path}")
            logger.info("Returning empty memory")
            return []
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load memory: {str(e)}")

    def _save_memory(self, data: List[Dict[str, Any]]) -> None:
        """
        Save memory data to the JSON file.

        Args:
            data (List[Dict[str, Any]]): Memory data to save.
            
        Raises:
            RuntimeError: If memory file cannot be saved.
        """
        try:
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
                logger.debug(f"Saved {len(data)} entries to memory file")
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to save memory: {str(e)}")

    def add_entry(self, task: str, feedback: str, embedding: List[float]) -> None:
        """
        Add a new feedback entry to memory.

        Args:
            task (str): The task description.
            feedback (str): The feedback for the task.
            embedding (List[float]): The embedding vector for the task.
            
        Raises:
            ValueError: If task, feedback, or embedding is invalid.
            RuntimeError: If memory entry cannot be added.
        """
        if not task or not feedback:
            logger.error("Task and feedback must not be empty")
            raise ValueError("Task and feedback must not be empty")
            
        if not embedding or not isinstance(embedding, list):
            logger.error("Embedding must be a non-empty list")
            raise ValueError("Embedding must be a non-empty list")
        
        try:
            logger.debug(f"Adding new entry for task: {task[:50]}{'...' if len(task) > 50 else ''}")
            memory = self._load_memory()
            
            # Create new entry with default usage count
            new_entry = {
                "task": task,
                "feedback": feedback,
                "embedding": embedding,
                "times_used": 0
            }
            
            memory.append(new_entry)
            self._save_memory(memory)
            logger.info(f"Added new feedback entry (total entries: {len(memory)})")
        except Exception as e:
            logger.error(f"Error adding memory entry: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to add memory entry: {str(e)}")

    def find_similar(self, embedding: List[float], threshold: float = 0.85, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Find similar tasks based on embedding similarity.

        Args:
            embedding (List[float]): The embedding to compare against.
            threshold (float): Minimum similarity threshold (0-1).
            top_k (int): Maximum number of results to return.

        Returns:
            List[Dict[str, Any]]: List of similar tasks with their feedback and similarity score.
            
        Raises:
            ValueError: If embedding is invalid or threshold is out of range.
            RuntimeError: If similarity search fails.
        """
        if not embedding or not isinstance(embedding, list):
            logger.error("Embedding must be a non-empty list")
            raise ValueError("Embedding must be a non-empty list")
            
        if threshold <= 0 or threshold > 1:
            logger.error(f"Similarity threshold must be between 0 and 1, got {threshold}")
            raise ValueError(f"Similarity threshold must be between 0 and 1, got {threshold}")
            
        if top_k < 1:
            logger.error(f"top_k must be at least 1, got {top_k}")
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        
        try:
            memory = self._load_memory()
            if not memory:
                logger.debug("Memory is empty, returning empty similar list")
                return []
            
            logger.debug(f"Finding similar tasks with threshold={threshold}, top_k={top_k}")
            
            # Convert embedding to numpy array for vector operations
            query_embedding = np.array(embedding)
            
            # Calculate similarity for each memory entry
            similarities = []
            for i, entry in enumerate(memory):
                memory_embedding = np.array(entry["embedding"])
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, memory_embedding)
                
                if similarity >= threshold:
                    similarities.append({
                        "index": i,
                        "similarity": similarity,
                        "task": entry["task"],
                        "feedback": entry["feedback"],
                        "times_used": entry.get("times_used", 0)
                    })
            
            # Sort by similarity (highest first) and take top k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            result = similarities[:top_k]
            
            logger.debug(f"Found {len(result)} similar tasks")
            return result
        except Exception as e:
            logger.error(f"Error finding similar tasks: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to find similar tasks: {str(e)}")

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            v1 (np.ndarray): First vector.
            v2 (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity (0-1).
        """
        try:
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            return dot_product / (norm_v1 * norm_v2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}", exc_info=True)
            return 0.0  # Return 0 similarity on error

    def increment_usage(self, indices: List[int]) -> None:
        """
        Increment the usage count for specific memory entries.

        Args:
            indices (List[int]): Indices of entries to increment.
            
        Raises:
            RuntimeError: If usage counts cannot be updated.
        """
        if not indices:
            logger.debug("No indices provided to increment_usage")
            return
        
        try:
            logger.debug(f"Incrementing usage counts for indices: {indices}")
            memory = self._load_memory()
            
            if not memory:
                logger.warning("Memory is empty, cannot increment usage counts")
                return
                
            updated = False
            for idx in indices:
                if 0 <= idx < len(memory):
                    memory[idx]["times_used"] = memory[idx].get("times_used", 0) + 1
                    updated = True
                else:
                    logger.warning(f"Index {idx} is out of range (0-{len(memory)-1})")
            
            if updated:
                self._save_memory(memory)
                logger.debug("Updated usage counts successfully")
        except Exception as e:
            logger.error(f"Error incrementing usage counts: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to increment usage counts: {str(e)}")

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all memory entries in a readable format (without embeddings).

        Returns:
            List[Dict[str, Any]]: All stored memory entries without embedding vectors.
        """
        try:
            memory = self._load_memory()
            
            # Exclude embeddings from output to make it more readable
            readable_memory = []
            for entry in memory:
                readable_entry = {
                    "task": entry["task"],
                    "feedback": entry["feedback"],
                    "times_used": entry.get("times_used", 0)
                }
                readable_memory.append(readable_entry)
            
            logger.debug(f"Retrieved {len(readable_memory)} memory entries")
            return readable_memory
        except Exception as e:
            logger.error(f"Error retrieving all memory entries: {str(e)}", exc_info=True)
            return []  # Return empty list on error

    def reset(self) -> None:
        """
        Reset the memory file to an empty state.
        
        Raises:
            RuntimeError: If memory cannot be reset.
        """
        try:
            logger.info(f"Resetting memory file: {self.file_path}")
            self._save_memory([])
        except Exception as e:
            logger.error(f"Error resetting memory: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to reset memory: {str(e)}")

    def delete(self) -> None:
        """
        Delete the memory file.
        
        Raises:
            RuntimeError: If memory file cannot be deleted.
        """
        try:
            if os.path.exists(self.file_path):
                logger.info(f"Deleting memory file: {self.file_path}")
                os.remove(self.file_path)
        except Exception as e:
            logger.error(f"Error deleting memory file: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to delete memory file: {str(e)}")

    def get_file_path(self) -> str:
        """
        Get the path to the memory file.

        Returns:
            str: Path to the memory file.
        """
        return self.file_path

    def is_empty(self) -> bool:
        """
        Check if the memory is empty.

        Returns:
            bool: True if memory is empty, False otherwise.
        """
        memory = self._load_memory()
        return len(memory) == 0

    def __del__(self):
        """
        Clean up temporary files when the object is deleted.
        """
        if hasattr(self, 'is_temporary') and self.is_temporary and hasattr(self, 'temp_dir'):
            try:
                if os.path.exists(self.temp_dir):
                    logger.debug(f"Cleaning up temporary directory: {self.temp_dir}")
                    shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory: {str(e)}") 