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
    
    # Define model name patterns for GBLLM backend
    GBLLM_PATTERNS = {
        r'channel-mix': LayerMode.CHANNEL_MIX,
        r'layer-mix': LayerMode.LAYER_MIX,
        r'B-(\d+)bit-groupsize(\d+)': LayerMode.LEGENCY,
        r'w(\d+)a\d+g(\d+)': LayerMode.LEGENCY,
    }
    
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        if model_name.endswith('-mlx'):
            raise ValueError("Models with '-mlx' suffix are not supported")
            
        self.backend_type = self._determine_backend_type(model_name)
        self.model = self._initialize_backend(model_name, **kwargs)
    
    def _determine_backend_type(self, model_name: str) -> BackendType:
        """Determine the appropriate backend type based on model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            BackendType: The determined backend type
        """
        for pattern in self.GBLLM_PATTERNS:
            if re.search(pattern, model_name):
                return BackendType.GBLLM
        return BackendType.VLLM
    
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
        