"""
SelfLearner module for enhancing prompts based on past feedback.

Manages task similarity, feedback selection, and prompt enhancement.
"""

import os
import json
import logging
import shutil
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from .embedder import Embedder
from .memory import Memory

# Configure module logger
logger = logging.getLogger(__name__)

class SelfLearner:
    """
    Main class that manages feedback memory and improves prompts.
    """

    def __init__(
        self,
        embedding_model: str = "miniLM",
        memory_path: str = "memory.json",
        similarity_threshold: float = 0.85,
        max_matches: int = 2,
        num_select_feedback: int = 1,
        llm_feedback_selection_layer: str = "openai",
        feedback_formatter: Optional[Callable[[str, str], str]] = None,
        temporary_memory: bool = False,
        clear_memory: bool = False,
        show_feedback_selection: bool = False
    ):
        """
        Initialize the SelfLearner.

        Args:
            embedding_model (str): Model to use for embeddings.
                                  Options: "openai", "miniLM", "bge-small"
                                  Defaults to "miniLM".
            memory_path (str): Path to the memory file.
                              Defaults to "memory.json".
                              Can specify a subdirectory like "project1/memory.json"
                              for domain-specific memory.
            similarity_threshold (float): Threshold for similarity matching.
                                         Defaults to 0.85.
            max_matches (int): Maximum number of similar tasks to retrieve.
                              Defaults to 2.
            num_select_feedback (int): Number of feedback items to select.
                                     If set to max_matches, all feedback will be selected.
                                     Defaults to 1 (best feedback only).
            llm_feedback_selection_layer (str): Provider for LLM feedback selection.
                               Options: "openai"
                               Defaults to "openai".
            feedback_formatter (Optional[Callable[[str, str], str]]): Custom function
                                                                     to format how feedback
                                                                     is injected into the prompt.
                                                                     Takes (base_prompt, feedback)
                                                                     and returns formatted prompt.
            temporary_memory (bool): If True, memory will be stored in a temporary 
                                    file that gets deleted when the program exits.
                                    Useful for testing or one-off runs.
                                    Defaults to False.
            clear_memory (bool): If True, any existing memory will be cleared
                                upon initialization.
                                Defaults to False.
            show_feedback_selection (bool): If True, logs information about 
                                           which feedback was selected and applied.
                                           Defaults to False.
        """
        logger.debug(f"Initializing SelfLearner with embedding_model={embedding_model}, "
                     f"memory_path={memory_path}, similarity_threshold={similarity_threshold}")
        
        # Validate inputs
        if similarity_threshold <= 0 or similarity_threshold > 1:
            raise ValueError(f"Similarity threshold must be between 0 and 1, got {similarity_threshold}")
        if max_matches < 1:
            raise ValueError(f"max_matches must be at least 1, got {max_matches}")
        if num_select_feedback < 1:
            raise ValueError(f"num_select_feedback must be at least 1, got {num_select_feedback}")
        if num_select_feedback > max_matches:
            logger.warning(f"num_select_feedback ({num_select_feedback}) is greater than max_matches ({max_matches}). Setting num_select_feedback to {max_matches}.")
            num_select_feedback = max_matches
            
        self.embedder = Embedder(model_name=embedding_model)
        self.memory = Memory(file_path=memory_path, temporary=temporary_memory)
        self.similarity_threshold = similarity_threshold
        self.max_matches = max_matches
        self.num_select_feedback = num_select_feedback
        self.llm_provider = llm_feedback_selection_layer.lower()  # Keep for backwards compatibility
        self.llm_feedback_selection_layer = llm_feedback_selection_layer.lower()
        self.feedback_formatter = feedback_formatter
        self.show_feedback_selection = show_feedback_selection
        self.memory_path = memory_path
        
        # Clear memory if requested
        if clear_memory and os.path.exists(self.memory.get_file_path()):
            logger.info(f"Clearing memory at {self.memory.get_file_path()}")
            self.memory.reset()
            
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize the LLM client based on the provider."""
        self.llm_client = None
        self.async_llm_client = None
        
        if self.llm_feedback_selection_layer == "openai":
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                try:
                    import openai
                    self.llm_client = openai.OpenAI(api_key=openai_key)
                    self.async_llm_client = openai.AsyncOpenAI(api_key=openai_key)
                    logger.info("Successfully initialized OpenAI client for feedback selection")
                except ImportError:
                    logger.warning("OpenAI package not installed. Feedback selection will be done without LLM.")
            else:
                logger.warning("OpenAI API key not found. Feedback selection will be done without LLM.")
        else:
            logger.warning(f"Unknown LLM provider: {self.llm_feedback_selection_layer}. "
                          "Feedback selection will be done without LLM.")
                          
    async def apply_feedback_async(self, task: str, base_prompt: str) -> str:
        """
        Asynchronously apply relevant feedback to a prompt based on task similarity.

        Args:
            task (str): The task description.
            base_prompt (str): The base prompt to enhance.

        Returns:
            str: Enhanced prompt with relevant feedback applied.
        """
        if not task:
            logger.warning("Empty task provided, returning base prompt unchanged")
            return base_prompt
        
        try:
            # Generate embedding for the task asynchronously
            task_embedding = await self.embedder.embed_async(task)
            
            # Find similar tasks in memory
            similar_tasks = self.memory.find_similar(
                embedding=task_embedding,
                threshold=self.similarity_threshold,
                top_k=self.max_matches
            )
            
            # If no similar tasks found, return the base prompt
            if not similar_tasks:
                if self.show_feedback_selection:
                    logger.info("No similar tasks found. Using original prompt.")
                return base_prompt
            
            # If showing feedback selection, display what was found
            if self.show_feedback_selection:
                logger.info(f"Found {len(similar_tasks)} similar tasks:")
                for i, task_info in enumerate(similar_tasks):
                    logger.info(f"{i+1}. Task: '{task_info['task']}'")
                    logger.info(f"   Similarity score: {task_info['similarity']:.4f}")
                    logger.info(f"   Feedback: '{task_info['feedback']}'")
            
            # If only one similar task found, use its feedback
            if len(similar_tasks) == 1:
                selected_feedback = similar_tasks[0]["feedback"]
                selected_task = similar_tasks[0]["task"]
                selected_index = similar_tasks[0]["index"]
                # Update usage count
                self.memory.increment_usage([selected_index])
                
                if self.show_feedback_selection:
                    logger.info(f"Selected feedback: '{selected_feedback}'")
                    logger.info(f"From task: '{selected_task}'")
            else:
                # If multiple similar tasks found, use LLM to select the best feedback asynchronously
                selected_feedback = await self._select_best_feedback_async(task, similar_tasks)
                
                # Handle different types of selected_feedback
                if isinstance(selected_feedback, list):
                    # Find which tasks provided the selected feedback
                    selected_indices = []
                    for feedback_item in selected_feedback:
                        if feedback_item == "NONE":
                            continue
                        for similar_task in similar_tasks:
                            if similar_task["feedback"] == feedback_item:
                                selected_indices.append(similar_task["index"])
                                break
                    
                    # Update usage count
                    if selected_indices:
                        self.memory.increment_usage(selected_indices)
                        
                    if self.show_feedback_selection:
                        logger.info(f"Selected {len(selected_feedback)} feedback items")
                        for item in selected_feedback:
                            logger.info(f"- '{item}'")
                else:
                    # Legacy single feedback selection (for backward compatibility)
                    # Find which task provided the selected feedback
                    selected_task = None
                    selected_index = None
                    for similar_task in similar_tasks:
                        if similar_task["feedback"] == selected_feedback:
                            selected_task = similar_task["task"]
                            selected_index = similar_task["index"]
                            # Update usage count
                            self.memory.increment_usage([selected_index])
                            break
                            
                    if self.show_feedback_selection and selected_task:
                        logger.info(f"Selected feedback: '{selected_feedback}'")
                        logger.info(f"From task: '{selected_task}'")
            
            # If no suitable feedback was selected, return the base prompt
            if selected_feedback == "NONE" or (isinstance(selected_feedback, list) and selected_feedback[0] == "NONE"):
                if self.show_feedback_selection:
                    logger.info("No suitable feedback was selected. Using original prompt.")
                return base_prompt
            
            # Enhance the prompt with the selected feedback
            enhanced_prompt = self._inject_feedback(base_prompt, selected_feedback)
            
            if self.show_feedback_selection:
                logger.debug("Original prompt:")
                logger.debug(f"'{base_prompt}'")
                logger.debug("Enhanced prompt:")
                logger.debug(f"'{enhanced_prompt}'")
                
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error applying feedback asynchronously: {str(e)}", exc_info=True)
            logger.warning("Returning original prompt due to error")
            return base_prompt
            
    async def enhance_prompt_async(self, task: str, base_prompt: str) -> str:
        """
        [DEPRECATED] Asynchronous version of enhance_prompt. Use apply_feedback_async instead.
        This method remains for backward compatibility.
        """
        logger.warning("enhance_prompt_async is deprecated, use apply_feedback_async instead")
        return await self.apply_feedback_async(task, base_prompt)
        
    async def _select_best_feedback_async(self, task: str, similar_tasks: List[Dict[str, Any]]) -> Union[str, List[str]]:
        """
        Asynchronously select the best feedback from similar tasks.
        
        If an async LLM client is available, use it to choose.
        Otherwise, use the most similar task's feedback.

        Args:
            task (str): The current task.
            similar_tasks (List[Dict[str, Any]]): List of similar tasks with feedback.

        Returns:
            Union[str, List[str]]: The selected feedback(s) or "NONE" if none is suitable.
                                 Returns a string if num_select_feedback=1, otherwise a list.
        """
        # If selecting all available feedback
        if self.num_select_feedback >= len(similar_tasks):
            logger.debug(f"Selecting all {len(similar_tasks)} available feedback items")
            return [task["feedback"] for task in similar_tasks]
        
        # If no async LLM client available, use the most similar tasks' feedback
        if not self.async_llm_client:
            logger.debug("No async LLM client available, using most similar tasks' feedback")
            return [task["feedback"] for task in similar_tasks[:self.num_select_feedback]]
        
        # Construct prompt for the LLM
        prompt = self._construct_feedback_selection_prompt(task, similar_tasks)
        
        if self.show_feedback_selection:
            logger.debug("Feedback selection prompt (async):")
            logger.debug(f"'{prompt}'")
        
        try:
            if self.llm_feedback_selection_layer == "openai":
                response = await self.async_llm_client.chat.completions.create(
                    model="gpt-4o",  # or gpt-3.5-turbo as a fallback
                    messages=[
                        {"role": "system", "content": "You are helping improve a self-learning agent."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                selected_feedback_text = response.choices[0].message.content.strip()
                
                if self.show_feedback_selection:
                    logger.debug(f"Async LLM response: '{selected_feedback_text}'")
                
                # Parse selected feedback(s)
                return self._parse_selected_feedback(selected_feedback_text, similar_tasks)
            else:
                logger.warning(f"Unknown LLM provider: {self.llm_feedback_selection_layer}, using most similar tasks' feedback")
                return [task["feedback"] for task in similar_tasks[:self.num_select_feedback]]
                
        except Exception as e:
            logger.error(f"Error selecting feedback with async LLM: {str(e)}", exc_info=True)
            logger.warning("Falling back to most similar task's feedback.")
            # Fall back to the most similar task's feedback
            return [task["feedback"] for task in similar_tasks[:self.num_select_feedback]]

    def _select_best_feedback(self, task: str, similar_tasks: List[Dict[str, Any]]) -> Union[str, List[str]]:
        """
        Select the best feedback from similar tasks.
        
        If an LLM client is available, use it to choose.
        Otherwise, use the most similar task's feedback.

        Args:
            task (str): The current task.
            similar_tasks (List[Dict[str, Any]]): List of similar tasks with feedback.

        Returns:
            Union[str, List[str]]: The selected feedback(s) or "NONE" if none is suitable.
                                 Returns a string if num_select_feedback=1, otherwise a list.
        """
        # If selecting all available feedback
        if self.num_select_feedback >= len(similar_tasks):
            logger.debug(f"Selecting all {len(similar_tasks)} available feedback items")
            return [task["feedback"] for task in similar_tasks]
        
        # If no LLM client available, use the most similar tasks' feedback
        if not self.llm_client:
            logger.debug("No LLM client available, using most similar tasks' feedback")
            return [task["feedback"] for task in similar_tasks[:self.num_select_feedback]]
        
        # Construct prompt for the LLM
        prompt = self._construct_feedback_selection_prompt(task, similar_tasks)
        
        if self.show_feedback_selection:
            logger.debug("Feedback selection prompt:")
            logger.debug(f"'{prompt}'")
        
        try:
            if self.llm_feedback_selection_layer == "openai":
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o",  # or gpt-3.5-turbo as a fallback
                    messages=[
                        {"role": "system", "content": "You are helping improve a self-learning agent."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                selected_feedback_text = response.choices[0].message.content.strip()
                
                if self.show_feedback_selection:
                    logger.debug(f"LLM response: '{selected_feedback_text}'")
                
                # Parse selected feedback(s)
                return self._parse_selected_feedback(selected_feedback_text, similar_tasks)
            else:
                logger.warning(f"Unknown LLM provider: {self.llm_feedback_selection_layer}, using most similar tasks' feedback")
                return [task["feedback"] for task in similar_tasks[:self.num_select_feedback]]
                
        except Exception as e:
            logger.error(f"Error selecting feedback with LLM: {str(e)}", exc_info=True)
            logger.warning("Falling back to most similar task's feedback.")
            # Fall back to the most similar task's feedback
            return [task["feedback"] for task in similar_tasks[:self.num_select_feedback]]

    def _parse_selected_feedback(self, response: str, similar_tasks: List[Dict[str, Any]]) -> Union[str, List[str]]:
        """
        Parse the LLM's response to extract selected feedback.
        
        Args:
            response (str): The LLM's response text.
            similar_tasks (List[Dict[str, Any]]): The list of similar tasks.
            
        Returns:
            Union[str, List[str]]: The selected feedback(s) or the most similar task's feedback.
        """
        # If only selecting one feedback item (backward compatibility)
        if self.num_select_feedback == 1:
            # Check for exact matches
            for task in similar_tasks:
                if response == task["feedback"]:
                    return response
                
            # If response is "NONE"
            if response == "NONE":
                return "NONE"
                
            # Otherwise default to most similar
            logger.warning("LLM response didn't match any feedback exactly. Using most similar task's feedback.")
            return similar_tasks[0]["feedback"]
        
        # For multiple feedback selection
        selected_feedbacks = []
        
        # Check for "NONE" response
        if response == "NONE":
            return ["NONE"]
            
        # Try to extract Feedback X: patterns
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Check for each feedback in the response
            for task in similar_tasks:
                if task["feedback"] in line:
                    if task["feedback"] not in selected_feedbacks:
                        selected_feedbacks.append(task["feedback"])
        
        # If no feedback was extracted from the response
        if not selected_feedbacks:
            logger.warning("Couldn't extract feedback from LLM response. Using top similar tasks' feedback.")
            selected_feedbacks = [task["feedback"] for task in similar_tasks[:self.num_select_feedback]]
            
        return selected_feedbacks

    def _construct_feedback_selection_prompt(self, task: str, similar_tasks: List[Dict[str, Any]]) -> str:
        """
        Construct a prompt for the LLM to select the best feedback.

        Args:
            task (str): The current task.
            similar_tasks (List[Dict[str, Any]]): List of similar tasks with feedback.

        Returns:
            str: The prompt for the LLM.
        """
        prompt = f"""You are helping improve a self-learning agent.

Here is a new incoming task: {task}

Here are previous feedbacks from similar tasks:
"""

        for i, similar_task in enumerate(similar_tasks):
            prompt += f"- Feedback {i+1}: {similar_task['feedback']}\n"

        # Adjust prompt based on num_select_feedback
        if self.num_select_feedback == 1:
            prompt += """
Choose the **most appropriate feedback** to apply to the new task.
Return the selected feedback text EXACTLY as it appears above.
If none are useful, return "NONE".

Selected feedback:"""
        else:
            prompt += f"""
Choose the **{self.num_select_feedback} most appropriate feedback items** to apply to the new task.
For each selected feedback, return the feedback text EXACTLY as it appears above, prefixed with "Feedback X: "
If none are useful, return "NONE".

Selected feedback items:"""

        return prompt

    def _inject_feedback(self, base_prompt: str, feedback: Union[str, List[str]]) -> str:
        """
        Inject feedback into the base prompt.
        
        Uses custom formatter if provided, otherwise uses default format.

        Args:
            base_prompt (str): The base prompt.
            feedback (Union[str, List[str]]): The feedback to inject. Can be a string or list of strings.

        Returns:
            str: The enhanced prompt with feedback injected.
        """
        # Handle case where feedback is a list
        if isinstance(feedback, list):
            if len(feedback) == 1:
                # If only one feedback item, treat as string
                return self._inject_feedback(base_prompt, feedback[0])
            
            # Combine multiple feedback items
            combined_feedback = "\n".join([f"- {f}" for f in feedback])
            
            if self.feedback_formatter:
                return self.feedback_formatter(base_prompt, combined_feedback)
            
            # Default formatting for multiple feedback items
            return f"{base_prompt}\n\nAdditional instructions:\n{combined_feedback}"
        
        # Handle string feedback (original behavior)
        if self.feedback_formatter:
            return self.feedback_formatter(base_prompt, feedback)
        
        # Default formatting: append to the end of the base prompt
        return f"{base_prompt}\n\nAdditional instructions: {feedback}"

    def save_feedback(self, task: str, feedback: str) -> None:
        """
        Save feedback for a task.

        Args:
            task (str): The task description.
            feedback (str): The feedback for the task.
        """
        if not task or not feedback:
            logger.error("Task and feedback cannot be empty")
            raise ValueError("Task and feedback cannot be empty")
            
        try:
            # Generate embedding for the task
            task_embedding = self.embedder.embed(task)
            
            # Save to memory
            self.memory.add_entry(task, feedback, task_embedding)
            logger.info(f"Saved feedback for task: '{task[:50]}...' if len(task) > 50 else task")
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}", exc_info=True)
            raise

    async def save_feedback_async(self, task: str, feedback: str) -> None:
        """
        Asynchronously save feedback for a task.

        Args:
            task (str): The task description.
            feedback (str): The feedback for the task.
        """
        if not task or not feedback:
            logger.error("Task and feedback cannot be empty")
            raise ValueError("Task and feedback cannot be empty")
            
        try:
            # Generate embedding for the task asynchronously
            task_embedding = await self.embedder.embed_async(task)
            
            # Save to memory
            self.memory.add_entry(task, feedback, task_embedding)
            logger.info(f"Saved feedback for task: '{task[:50]}...' if len(task) > 50 else task")
        except Exception as e:
            logger.error(f"Error saving feedback asynchronously: {str(e)}", exc_info=True)
            raise

    def apply_feedback(self, task: str, base_prompt: str) -> str:
        """
        Apply relevant feedback to a prompt based on task similarity.

        Args:
            task (str): The task description.
            base_prompt (str): The base prompt to enhance.

        Returns:
            str: Enhanced prompt with relevant feedback applied.
        """
        if not task:
            logger.warning("Empty task provided, returning base prompt unchanged")
            return base_prompt
        
        try:
            # Generate embedding for the task
            task_embedding = self.embedder.embed(task)
            
            # Find similar tasks in memory
            similar_tasks = self.memory.find_similar(
                embedding=task_embedding,
                threshold=self.similarity_threshold,
                top_k=self.max_matches
            )
            
            # If no similar tasks found, return the base prompt
            if not similar_tasks:
                if self.show_feedback_selection:
                    logger.info("No similar tasks found. Using original prompt.")
                return base_prompt
            
            # If showing feedback selection, display what was found
            if self.show_feedback_selection:
                logger.info(f"Found {len(similar_tasks)} similar tasks:")
                for i, task_info in enumerate(similar_tasks):
                    logger.info(f"{i+1}. Task: '{task_info['task']}'")
                    logger.info(f"   Similarity score: {task_info['similarity']:.4f}")
                    logger.info(f"   Feedback: '{task_info['feedback']}'")
            
            # If only one similar task found, use its feedback
            if len(similar_tasks) == 1:
                selected_feedback = similar_tasks[0]["feedback"]
                selected_task = similar_tasks[0]["task"]
                selected_index = similar_tasks[0]["index"]
                # Update usage count
                self.memory.increment_usage([selected_index])
                
                if self.show_feedback_selection:
                    logger.info(f"Selected feedback: '{selected_feedback}'")
                    logger.info(f"From task: '{selected_task}'")
            else:
                # If multiple similar tasks found, use LLM to select the best feedback
                selected_feedback = self._select_best_feedback(task, similar_tasks)
                
                # Handle different types of selected_feedback
                if isinstance(selected_feedback, list):
                    # Find which tasks provided the selected feedback
                    selected_indices = []
                    for feedback_item in selected_feedback:
                        if feedback_item == "NONE":
                            continue
                        for similar_task in similar_tasks:
                            if similar_task["feedback"] == feedback_item:
                                selected_indices.append(similar_task["index"])
                                break
                    
                    # Update usage count
                    if selected_indices:
                        self.memory.increment_usage(selected_indices)
                        
                    if self.show_feedback_selection:
                        logger.info(f"Selected {len(selected_feedback)} feedback items")
                        for item in selected_feedback:
                            logger.info(f"- '{item}'")
                else:
                    # Legacy single feedback selection (for backward compatibility)
                    # Find which task provided the selected feedback
                    selected_task = None
                    selected_index = None
                    for similar_task in similar_tasks:
                        if similar_task["feedback"] == selected_feedback:
                            selected_task = similar_task["task"]
                            selected_index = similar_task["index"]
                            # Update usage count
                            self.memory.increment_usage([selected_index])
                            break
                            
                    if self.show_feedback_selection and selected_task:
                        logger.info(f"Selected feedback: '{selected_feedback}'")
                        logger.info(f"From task: '{selected_task}'")
            
            # If no suitable feedback was selected, return the base prompt
            if selected_feedback == "NONE" or (isinstance(selected_feedback, list) and selected_feedback[0] == "NONE"):
                if self.show_feedback_selection:
                    logger.info("No suitable feedback was selected. Using original prompt.")
                return base_prompt
            
            # Enhance the prompt with the selected feedback
            enhanced_prompt = self._inject_feedback(base_prompt, selected_feedback)
            
            if self.show_feedback_selection:
                logger.debug("Original prompt:")
                logger.debug(f"'{base_prompt}'")
                logger.debug("Enhanced prompt:")
                logger.debug(f"'{enhanced_prompt}'")
                
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error applying feedback: {str(e)}", exc_info=True)
            logger.warning("Returning original prompt due to error")
            return base_prompt
    
    # Keep enhance_prompt for backward compatibility
    def enhance_prompt(self, task: str, base_prompt: str) -> str:
        """
        [DEPRECATED] Use apply_feedback instead. 
        This method remains for backward compatibility.
        """
        logger.warning("enhance_prompt is deprecated, use apply_feedback instead")
        return self.apply_feedback(task, base_prompt)

    def show_memory(self) -> List[Dict[str, Any]]:
        """
        Show all memory entries.

        Returns:
            List[Dict[str, Any]]: All memory entries.
        """
        return self.memory.get_all()

    def list_all_feedback(self, memory_path: str = None, verbose: bool = False) -> None:
        """
        List all feedback entries in memory.

        Args:
            memory_path (str, optional): Path to the memory file.
                                         Defaults to None, which uses the instance's memory_path.
            verbose (bool, optional): Whether to print detailed output. Defaults to False.
        """
        try:
            # Use provided memory path or default
            memory_path = memory_path or self.memory_path
            temp_memory = Memory(file_path=memory_path)
            entries = temp_memory.get_all()
            
            if not entries:
                logger.info(f"No feedback found in {memory_path}")
                return
                
            logger.info(f"Found {len(entries)} feedback entries in {memory_path}:")
            
            for i, entry in enumerate(entries):
                # Truncate task if too long
                task_display = entry["task"]
                if len(task_display) > 50:
                    task_display = task_display[:50] + "..."
                    
                # Format differently based on verbosity
                if verbose:
                    logger.info(f"{i}. Task: '{task_display}'")
                    logger.info(f"   Feedback: '{entry['feedback']}'")
                    logger.info(f"   Used {entry.get('times_used', 0)} times")
                else:
                    logger.info(f"{i}. '{task_display}' - '{entry['feedback']}' (used {entry.get('times_used', 0)} times)")
        except Exception as e:
            logger.error(f"Error listing feedback: {str(e)}", exc_info=True)
            raise

    def list_feedback(self, task: str, memory_path: str = None, verbose: bool = False) -> None:
        """
        List feedback for a specific task.

        Args:
            task (str): The task to find feedback for.
            memory_path (str, optional): Path to the memory file.
                                         Defaults to None, which uses the instance's memory_path.
            verbose (bool, optional): Whether to print detailed output. Defaults to False.
        """
        try:
            if not task:
                logger.error("Task cannot be empty")
                raise ValueError("Task cannot be empty")
                
            # Use provided memory path or default
            memory_path = memory_path or self.memory_path
            
            # Use a temporary embedder to generate embedding for the task
            task_embedding = self.embedder.embed(task)
            
            # Use a temporary memory to find similar tasks
            temp_memory = Memory(file_path=memory_path)
            similar = temp_memory.find_similar(
                embedding=task_embedding,
                threshold=0.90,  # Higher threshold for exact matches
                top_k=10  # Get more potential matches
            )
            
            # Filter for exact matches
            exact_matches = [t for t in similar if t["task"] == task]
            similar = [t for t in similar if t["task"] != task]
            
            if not exact_matches and not similar:
                logger.info(f"No feedback found for task: '{task}'")
                return
                
            # Show exact matches first
            if exact_matches:
                logger.info(f"Found {len(exact_matches)} exact match(es) for task: '{task}'")
                for i, entry in enumerate(exact_matches):
                    if verbose:
                        logger.info(f"{i}. Feedback: '{entry['feedback']}'")
                        logger.info(f"   Used {entry.get('times_used', 0)} times")
                    else:
                        logger.info(f"{i}. '{entry['feedback']}' (used {entry.get('times_used', 0)} times)")
            
            # Then show similar matches
            if similar and verbose:
                logger.info(f"\nFound {len(similar)} similar task(s):")
                for i, entry in enumerate(similar):
                    logger.info(f"{i}. Task: '{entry['task']}'")
                    logger.info(f"   Similarity: {entry['similarity']:.4f}")
                    logger.info(f"   Feedback: '{entry['feedback']}'")
                    logger.info(f"   Used {entry.get('times_used', 0)} times")
        except Exception as e:
            logger.error(f"Error listing feedback for task: {str(e)}", exc_info=True)
            raise

    def list_feedback_substring(self, task_substring: str, memory_path: str = None, verbose: bool = False) -> None:
        """
        List feedback for tasks containing a substring.

        Args:
            task_substring (str): The substring to search for in tasks.
            memory_path (str, optional): Path to the memory file.
                                         Defaults to None, which uses the instance's memory_path.
            verbose (bool, optional): Whether to print detailed output. Defaults to False.
        """
        try:
            if not task_substring:
                logger.error("Search substring cannot be empty")
                raise ValueError("Search substring cannot be empty")
                
            # Use provided memory path or default
            memory_path = memory_path or self.memory_path
            
            # Use a temporary memory to get all entries
            temp_memory = Memory(file_path=memory_path)
            entries = temp_memory.get_all()
            
            # Filter for tasks containing the substring
            matches = [entry for entry in entries if task_substring.lower() in entry["task"].lower()]
            
            if not matches:
                logger.info(f"No tasks found containing '{task_substring}'")
                return
                
            logger.info(f"Found {len(matches)} task(s) containing '{task_substring}':")
            
            for i, entry in enumerate(matches):
                # Truncate task if too long
                task_display = entry["task"]
                if len(task_display) > 50:
                    task_display = task_display[:50] + "..."
                    
                # Format differently based on verbosity
                if verbose:
                    logger.info(f"{i}. Task: '{task_display}'")
                    logger.info(f"   Feedback: '{entry['feedback']}'")
                    logger.info(f"   Used {entry.get('times_used', 0)} times")
                else:
                    logger.info(f"{i}. '{task_display}' - '{entry['feedback']}' (used {entry.get('times_used', 0)} times)")
        except Exception as e:
            logger.error(f"Error listing feedback with substring: {str(e)}", exc_info=True)
            raise

    def remove_feedback(self, index: int = None, task_substring: str = None, memory_path: str = None) -> bool:
        """
        Remove feedback entry by index or containing a substring.
        
        Either index or task_substring must be provided.

        Args:
            index (int, optional): Index of the entry to remove.
            task_substring (str, optional): Remove entries containing this substring.
            memory_path (str, optional): Path to the memory file.
                                         Defaults to None, which uses the instance's memory_path.

        Returns:
            bool: True if successful, False otherwise.
            
        Raises:
            ValueError: If neither index nor task_substring is provided.
        """
        try:
            if index is None and not task_substring:
                logger.error("Either index or task_substring must be provided")
                raise ValueError("Either index or task_substring must be provided")
                
            # Use provided memory path or default
            memory_path = memory_path or self.memory_path
            
            # Use a temporary memory instance
            temp_memory = Memory(file_path=memory_path)
            memory_data = temp_memory._load_memory()
            
            if not memory_data:
                logger.warning(f"No feedback found in {memory_path}")
                return False
                
            # Remove by index
            if index is not None:
                if 0 <= index < len(memory_data):
                    removed_entry = memory_data.pop(index)
                    temp_memory._save_memory(memory_data)
                    
                    # Get truncated task for logging
                    task_display = removed_entry["task"]
                    if len(task_display) > 50:
                        task_display = task_display[:50] + "..."
                        
                    logger.info(f"Removed feedback for task: '{task_display}'")
                    return True
                else:
                    logger.warning(f"Invalid index: {index}, valid range is 0-{len(memory_data)-1}")
                    return False
            
            # Remove by substring
            elif task_substring:
                # Find entries containing the substring
                matches = [i for i, entry in enumerate(memory_data) 
                          if task_substring.lower() in entry["task"].lower()]
                
                if not matches:
                    logger.warning(f"No tasks found containing '{task_substring}'")
                    return False
                
                # If multiple matches, list them and return False
                if len(matches) > 1:
                    logger.warning(f"Multiple tasks ({len(matches)}) contain '{task_substring}'.")
                    logger.warning("Please specify an index to remove or use a more specific substring.")
                    
                    # List the matching entries
                    for i in matches:
                        # Get truncated task for logging
                        task_display = memory_data[i]["task"]
                        if len(task_display) > 50:
                            task_display = task_display[:50] + "..."
                            
                        logger.info(f"{i}. '{task_display}'")
                    
                    return False
                
                # If only one match, remove it
                removed_entry = memory_data.pop(matches[0])
                temp_memory._save_memory(memory_data)
                
                # Get truncated task for logging
                task_display = removed_entry["task"]
                if len(task_display) > 50:
                    task_display = task_display[:50] + "..."
                    
                logger.info(f"Removed feedback for task: '{task_display}'")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error removing feedback: {str(e)}", exc_info=True)
            return False

    def remove_feedback_for_task(self, task: str, memory_path: str = None) -> bool:
        """
        Remove all feedback entries for a specific task.

        Args:
            task (str): The task to remove feedback for.
            memory_path (str, optional): Path to the memory file.
                                         Defaults to None, which uses the instance's memory_path.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not task:
                logger.error("Task cannot be empty")
                raise ValueError("Task cannot be empty")
                
            # Use provided memory path or default
            memory_path = memory_path or self.memory_path
            
            # Use a temporary memory instance
            temp_memory = Memory(file_path=memory_path)
            memory_data = temp_memory._load_memory()
            
            if not memory_data:
                logger.warning(f"No feedback found in {memory_path}")
                return False
                
            # Find entries with matching task
            matches = [i for i, entry in enumerate(memory_data) if entry["task"] == task]
            
            if not matches:
                logger.warning(f"No exact task matches found for: '{task}'")
                return False
                
            # Remove all matching entries
            # Iterate in reverse to avoid index issues
            for i in sorted(matches, reverse=True):
                memory_data.pop(i)
                
            temp_memory._save_memory(memory_data)
            
            # Get truncated task for logging
            task_display = task
            if len(task_display) > 50:
                task_display = task_display[:50] + "..."
                
            logger.info(f"Removed {len(matches)} feedback entries for task: '{task_display}'")
            return True
        except Exception as e:
            logger.error(f"Error removing feedback for task: {str(e)}", exc_info=True)
            return False

    def reset_memory(self, permanent: bool = False) -> None:
        """
        Reset the memory, removing all feedback.

        Args:
            permanent (bool, optional): Whether to permanently delete the memory file.
                                       If False, creates an empty memory file.
                                       Defaults to False.
        """
        try:
            if permanent:
                self.memory.delete()
                logger.info(f"Permanently deleted memory file: {self.memory.get_file_path()}")
            else:
                self.memory.reset()
                logger.info(f"Reset memory file: {self.memory.get_file_path()}")
        except Exception as e:
            logger.error(f"Error resetting memory: {str(e)}", exc_info=True)
            raise

    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set the similarity threshold for matching tasks.

        Args:
            threshold (float): Threshold value between 0 and 1.
            
        Raises:
            ValueError: If threshold is not between 0 and 1.
        """
        if threshold <= 0 or threshold > 1:
            logger.error(f"Similarity threshold must be between 0 and 1, got {threshold}")
            raise ValueError(f"Similarity threshold must be between 0 and 1, got {threshold}")
            
        self.similarity_threshold = threshold
        logger.debug(f"Set similarity threshold to {threshold}")

    def set_max_matches(self, max_matches: int) -> None:
        """
        Set the maximum number of similar tasks to retrieve.

        Args:
            max_matches (int): Maximum number of matches.
            
        Raises:
            ValueError: If max_matches is less than 1.
        """
        if max_matches < 1:
            logger.error(f"max_matches must be at least 1, got {max_matches}")
            raise ValueError(f"max_matches must be at least 1, got {max_matches}")
            
        self.max_matches = max_matches
        logger.debug(f"Set max_matches to {max_matches}")

    def set_feedback_formatter(self, formatter: Callable[[str, str], str]) -> None:
        """
        Set the feedback formatter function.

        Args:
            formatter (Callable[[str, str], str]): Function to format feedback into prompts.
        """
        self.feedback_formatter = formatter
        logger.debug("Set custom feedback formatter")

    def set_transparency(self, show_feedback_selection: bool) -> None:
        """
        Set whether to show feedback selection process.

        Args:
            show_feedback_selection (bool): Whether to show feedback selection.
        """
        self.show_feedback_selection = show_feedback_selection
        logger.debug(f"Set show_feedback_selection to {show_feedback_selection}")

    def export_memory(self, file_path: str) -> None:
        """
        Export memory to a file.

        Args:
            file_path (str): Path to export the memory to.
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Copy the memory file
            shutil.copyfile(self.memory.get_file_path(), file_path)
            logger.info(f"Exported memory to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting memory: {str(e)}", exc_info=True)
            raise

    def import_memory(self, file_path: str, merge: bool = False) -> None:
        """
        Import memory from a file.

        Args:
            file_path (str): Path to import the memory from.
            merge (bool, optional): Whether to merge with existing memory.
                                  If False, replaces existing memory.
                                  Defaults to False.
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Import file does not exist: {file_path}")
                raise FileNotFoundError(f"Import file does not exist: {file_path}")
                
            if not merge:
                # Simply copy the file
                shutil.copyfile(file_path, self.memory.get_file_path())
                logger.info(f"Imported memory from {file_path} (replaced existing memory)")
            else:
                # Merge the memories
                # Load both memories
                with open(file_path, 'r') as f:
                    import_data = json.load(f)
                    
                existing_data = self.memory._load_memory()
                
                # Add imported entries to existing data
                existing_data.extend(import_data)
                
                # Save merged data
                self.memory._save_memory(existing_data)
                logger.info(f"Imported memory from {file_path} (merged with existing memory)")
        except Exception as e:
            logger.error(f"Error importing memory: {str(e)}", exc_info=True)
            raise

    def get_memory_path(self) -> str:
        """
        Get the path to the memory file.

        Returns:
            str: Path to the memory file.
        """
        return self.memory.get_file_path()

    def is_memory_temporary(self) -> bool:
        """
        Check if the memory is temporary.

        Returns:
            bool: True if memory is temporary, False otherwise.
        """
        return self.memory.is_temporary

    def is_memory_empty(self) -> bool:
        """
        Check if the memory is empty.

        Returns:
            bool: True if memory is empty, False otherwise.
        """
        return self.memory.is_empty()

    def set_num_select_feedback(self, num_select_feedback: int) -> None:
        """
        Set the number of feedback items to select.

        Args:
            num_select_feedback (int): Number of feedback items to select.
            
        Raises:
            ValueError: If num_select_feedback is less than 1.
        """
        if num_select_feedback < 1:
            logger.error(f"num_select_feedback must be at least 1, got {num_select_feedback}")
            raise ValueError(f"num_select_feedback must be at least 1, got {num_select_feedback}")
            
        if num_select_feedback > self.max_matches:
            logger.warning(f"num_select_feedback ({num_select_feedback}) is greater than max_matches ({self.max_matches}). Setting num_select_feedback to {self.max_matches}.")
            num_select_feedback = self.max_matches
            
        self.num_select_feedback = num_select_feedback
        logger.debug(f"Set num_select_feedback to {num_select_feedback}") 