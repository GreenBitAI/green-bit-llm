from typing import Optional, Dict, Any, Union
from backends.base import BaseInferenceBackend
from backends.green_bit_backend import GBLLMInferenceBackend
from backends.vllm_backend import VLLMInferenceBackend
from green_bit_llm.common.enum import LayerMode
from enum import Enum, auto
import re
from typing import List

class BackendType(Enum):
    """Enumeration of supported backend types."""
    GBLLM = auto()
    VLLM = auto()

class LLMInference:
    """Large Language Model Inference wrapper class.
    
    This class provides a unified interface for different LLM backends,
    automatically selecting the appropriate backend based on the model name.
    
    Args:
        model_name (str): Name of the model to load
        **kwargs: Additional arguments passed to the backend constructor
    
    Raises:
        ValueError: If model name ends with '-mlx' or if backend initialization fails
    """
    
    def __init__(self, model_name: str, backend_type: Optional[BackendType] = BackendType.GBLLM, **kwargs: Any) -> None:
        if backend_type not in [BackendType.GBLLM, BackendType.VLLM]:
            raise ValueError(f"Unsupported backend type: {backend_type}")
            
        self.backend_type = backend_type
        self.model = self._initialize_backend(model_name, **kwargs)
    
    def _initialize_backend(self, model_name: str, **kwargs: Any) -> BaseInferenceBackend:
        """Initialize the appropriate backend based on backend type.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for backend initialization
            
        Returns:
            BaseInferenceBackend: Initialized backend instance
            
        Raises:
            ValueError: If backend initialization fails
        """
        try:
            if self.backend_type == BackendType.GBLLM:
                return GBLLMInferenceBackend(model_name, **kwargs)
            return VLLMInferenceBackend(model_name, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to initialize {self.backend_type.name} backend: {str(e)}")
    
    def generate(self, 
                prompt: Union[str, List[str]], 
                params: Optional[Dict[str, Any]] = None) -> Any:
        """Generate text based on the input prompt.
        
        Args:
            prompt: Input prompt or list of prompts
            params: Optional generation parameters
            
        Returns:
            Generated text output from the model
            
        Raises:
            ValueError: If generation fails
        """
        try:
            if self.backend_type == BackendType.GBLLM:
                self.model.generate(prompt, params or {})
            else:
                self.model.generate(prompt, params)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")