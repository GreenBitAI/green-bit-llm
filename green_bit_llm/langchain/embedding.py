from typing import List, Dict, Any, Optional
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

# good balance between performance and efficiency, params: 22M
DEFAULT_MODEL_NAME1 = "sentence-transformers/all-MiniLM-L12-v2"
# params: 110M, better NLU ability
DEFAULT_MODEL_NAME2 = "sentence-transformers/all-mpnet-base-v2"
# Optimized for multi-round question answering and suitable for applications
# that require more complex context understanding.
DEFAULT_MODEL_NAME3 = "sentence-transformers/multi-qa-MiniLM-L6-cos-v"


class GreenBitEmbeddings(BaseModel, Embeddings):
    """GreenBit embedding model using sentence-transformers.

    This class provides an interface to generate embeddings using GreenBit's models,
    which are based on the sentence-transformers package.

    Attributes:
        model (Any): Embedding model.
        encode_kwargs (Dict[str, Any]): Additional keyword arguments for the encoding process.
        device (str): The device to use for computations (e.g., 'cuda' for GPU).

    Example:
        .. code-block:: python
        from reen_bit_llm.langchain import GreenBitEmbeddings

        embedder = GreenBitEmbeddings.from_model_id(
            "sentence-transformers/all-mpnet-base-v2",
            device="cuda",
            multi_process=True,
            model_kwargs={"cache_folder": "/path/to/cache"},
            encode_kwargs={"normalize_embeddings": True}
        )

        texts = ["Hello, world!", "This is a test."]
        document_embeddings = embedder.embed_documents(texts)
        query_embedding = embedder.embed_query("What is the meaning of life?")
    """
    cache_dir: Optional[str] = "~/.cache/huggingface/hub"
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the Sentence
    Transformer model, such as `prompt_name`, `prompt`, `batch_size`, `precision`,
    `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    show_progress: bool = False
    """Whether to show a progress bar."""
    device: str = "cuda"
    model: Any = None

    def __init__(self, **data):
        super().__init__(**data)

    class Config:
        """Configuration for this pydantic object."""

        extra = "forbid"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: The list of embeddings, one for each input text.
        """
        import sentence_transformers
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.model.encode(
                texts, show_progress_bar=self.show_progress, **self.encode_kwargs
            )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: The embedding for the input text.
        """
        return self.embed_documents([text])[0]

    @classmethod
    def from_model_id(
            cls,
            model_name: str = DEFAULT_MODEL_NAME1,
            device: str = "cuda",
            cache_dir: Optional[str] = "",
            multi_process: bool = False,
            show_progress: bool = False,
            model_kwargs: Dict[str, Any] = Field(default_factory=dict),
            encode_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict),
            **kwargs
    ) -> "GreenBitEmbeddings":
        """
        Create a GreenBitEmbeddings instance from a model name.

        Args:
            model_name (str): The name of the model to use.
            device (str): The device to use for computations (default is "cuda" for GPU).
            cache_dir (Optional[str]): Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable.
            multi_process (bool): Run encode() on multiple GPUs.
            show_progress (bool): Whether to show a progress bar.
            model_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to the Sentence Transformer model, such as `device`,
            `prompts`, `default_prompt_name`, `revision`, `trust_remote_code`, or `token`.
            See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer
            encode_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass when calling the `encode` method of the SentenceTransformer model, such as `prompt_name`, `prompt`, `batch_size`, `precision`,
            `normalize_embeddings`, and more. See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
            **kwargs: Additional keyword arguments for the GreenBitEmbeddings constructor.

        Returns:
            GreenBitEmbeddings: An instance of GreenBitEmbeddings.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers. "
                "Please install it with `pip install sentence-transformers`."
            )

        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir,
            **model_kwargs
        )

        return cls(
            model=model,
            device=device,
            cache_dir=cache_dir,
            multi_process=multi_process,
            show_progress=show_progress,
            encode_kwargs=encode_kwargs or {},
            **kwargs
        )