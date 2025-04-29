"""
Embedder module for generating embeddings from text using different models.

Supports:
- OpenAI (text-embedding-ada-002)
- MiniLM (sentence-transformers/all-MiniLM-L6-v2)
- BGE-small (BAAI/bge-small-en)
"""

import os
import logging
import numpy as np
from typing import List, Union, Optional
import asyncio

# Configure module logger
logger = logging.getLogger(__name__)

class Embedder:
    """
    A class for generating text embeddings using different models.
    Supports OpenAI API and local HuggingFace models.
    """

    def __init__(self, model_name: str = "miniLM"):
        """
        Initialize the Embedder with the specified model.

        Args:
            model_name (str): The name of the embedding model to use.
                              Options: "openai", "miniLM", "bge-small"
                              Defaults to "miniLM".
        """
        self.model_name = model_name.lower()
        self.model = None
        self.openai_client = None
        self.async_openai_client = None
        
        logger.debug(f"Initializing Embedder with model: {model_name}")
        
        # Initialize model based on selection
        if self.model_name == "openai":
            # Try to use OpenAI, fall back to HuggingFace if no API key
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                try:
                    import openai
                    self.openai_client = openai.OpenAI(api_key=openai_key)
                    self.async_openai_client = openai.AsyncOpenAI(api_key=openai_key)
                    logger.info("Successfully initialized OpenAI embeddings client")
                except ImportError:
                    logger.warning("OpenAI package not installed. Falling back to MiniLM.")
                    self.model_name = "miniLM"
            else:
                logger.warning("OpenAI API key not found. Falling back to MiniLM.")
                self.model_name = "miniLM"
        
        # If using a HuggingFace model (either by choice or fallback)
        if self.model_name in ["minilm", "bge-small"]:
            try:
                from sentence_transformers import SentenceTransformer
                
                model_mapping = {
                    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
                    "bge-small": "BAAI/bge-small-en"
                }
                
                model_path = model_mapping[self.model_name]
                logger.info(f"Loading embedding model: {model_path}")
                self.model = SentenceTransformer(model_path)
                logger.info(f"Successfully initialized {self.model_name} embeddings model")
            except ImportError:
                logger.error("sentence-transformers package not installed")
                raise ImportError(
                    "sentence-transformers package not installed. "
                    "Please install with: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Error loading model {model_path}: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to load embedding model: {str(e)}")

    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
            
        Raises:
            ValueError: If text is empty.
            RuntimeError: If embedding generation fails.
        """
        if not text:
            logger.error("Cannot embed empty text")
            raise ValueError("Text cannot be empty")
        
        try:
            logger.debug(f"Generating embedding for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            if self.model_name == "openai" and self.openai_client:
                # Generate embeddings using OpenAI API
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                logger.debug(f"Generated OpenAI embedding with dimension {len(embedding)}")
                return embedding
            elif self.model:
                # Generate embeddings using HuggingFace model
                embedding = self.model.encode([text])[0]
                embedding_list = embedding.tolist()
                logger.debug(f"Generated {self.model_name} embedding with dimension {len(embedding_list)}")
                return embedding_list
            else:
                logger.error("No embedding model available")
                raise RuntimeError("No embedding model available")
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    async def embed_async(self, text: str) -> List[float]:
        """
        Asynchronously generate an embedding vector for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
            
        Raises:
            ValueError: If text is empty.
            RuntimeError: If embedding generation fails.
        """
        if not text:
            logger.error("Cannot embed empty text")
            raise ValueError("Text cannot be empty")
        
        try:
            logger.debug(f"Generating async embedding for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            if self.model_name == "openai" and self.async_openai_client:
                # Generate embeddings using OpenAI API asynchronously
                response = await self.async_openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                logger.debug(f"Generated async OpenAI embedding with dimension {len(embedding)}")
                return embedding
            elif self.model:
                # Generate embeddings using HuggingFace model
                # Use asyncio to prevent blocking when used in async contexts
                embedding = await asyncio.to_thread(self.model.encode, [text])
                embedding_list = embedding[0].tolist()
                logger.debug(f"Generated async {self.model_name} embedding with dimension {len(embedding_list)}")
                return embedding_list
            else:
                logger.error("No embedding model available")
                raise RuntimeError("No embedding model available")
        except Exception as e:
            logger.error(f"Error generating async embedding: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate async embedding: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If embedding generation fails.
        """
        if not texts:
            logger.warning("Empty texts list provided to embed_batch")
            return []
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            if self.model_name == "openai" and self.openai_client:
                # Generate embeddings using OpenAI API
                response = self.openai_client.embeddings.create(
                    input=texts,
                    model="text-embedding-ada-002"
                )
                embeddings = [item.embedding for item in response.data]
                logger.debug(f"Generated {len(embeddings)} OpenAI embeddings")
                return embeddings
            elif self.model:
                # Generate embeddings using HuggingFace model
                embeddings = self.model.encode(texts)
                embeddings_list = embeddings.tolist()
                logger.debug(f"Generated {len(embeddings_list)} {self.model_name} embeddings")
                return embeddings_list
            else:
                logger.error("No embedding model available")
                raise RuntimeError("No embedding model available")
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generate embedding vectors for a batch of texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If embedding generation fails.
        """
        if not texts:
            logger.warning("Empty texts list provided to embed_batch_async")
            return []
        
        try:
            logger.debug(f"Generating async embeddings for {len(texts)} texts")
            
            if self.model_name == "openai" and self.async_openai_client:
                # Generate embeddings using OpenAI API asynchronously
                response = await self.async_openai_client.embeddings.create(
                    input=texts,
                    model="text-embedding-ada-002"
                )
                embeddings = [item.embedding for item in response.data]
                logger.debug(f"Generated {len(embeddings)} async OpenAI embeddings")
                return embeddings
            elif self.model:
                # Generate embeddings using HuggingFace model in a non-blocking way
                embeddings = await asyncio.to_thread(self.model.encode, texts)
                embeddings_list = embeddings.tolist()
                logger.debug(f"Generated {len(embeddings_list)} async {self.model_name} embeddings")
                return embeddings_list
            else:
                logger.error("No embedding model available")
                raise RuntimeError("No embedding model available")
        except Exception as e:
            logger.error(f"Error generating async batch embeddings: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate async batch embeddings: {str(e)}") 