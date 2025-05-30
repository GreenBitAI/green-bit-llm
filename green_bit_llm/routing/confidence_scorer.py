import torch
from green_bit_llm.routing.libra_router.ue_router import MahalanobisDistanceSeq


class ConfidenceScorer:
    """
    A class to compute confidence scores based on Mahalanobis distance.

    Attributes:
        parameters_path (str): Path to the model parameters
        json_file_path (str): Path to the uncertainty bounds JSON file
        device (str): Device to run computations on ('cpu', 'cuda', 'mps')

    Example:
        confidence_scorer = ConfidenceScore(
            parameters_path="path/to/params",
            json_file_path="path/to/bounds.json",
            device="cuda"
        )

        confidence = confidence_scorer.calculate_confidence(hidden_states)
    """

    def __init__(
            self,
            parameters_path: str,
            model_id: str,
            device: str = "cuda"
    ):
        """
        Initialize the ConfidenceScore calculator.

        Args:
            parameters_path: Path to model parameters
            json_file_path: Path to uncertainty bounds JSON
            device: Computation device
            threshold: Confidence threshold for routing
        """
        self.parameters_path = parameters_path
        self.device = device

        # Initialize Mahalanobis distance calculator
        try:
            self.mahalanobis = MahalanobisDistanceSeq(
                parameters_path=parameters_path,
                normalize=False,
                model_id=model_id,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mahalanobis distance calculator: {str(e)}")

    def calculate_confidence(
            self,
            hidden_states: torch.Tensor
    ) -> float:
        """
        Calculate confidence score from hidden states.
        Support both single input and batch input.

        Args:
            hidden_states: Model hidden states tensor
            return_uncertainty: Whether to return the raw uncertainty score

        Returns:
            Union[float, List[float]]: Single confidence score or list of confidence scores

        Raises:
            ValueError: If hidden states have invalid shape
            RuntimeError: If confidence calculation fails
        """

        try:
            # Calculate uncertainty using Mahalanobis distance
            uncertainty = self.mahalanobis(hidden_states)
            if uncertainty is None:
                raise RuntimeError("Failed to calculate uncertainty")

            # Normalize uncertainty if bounds are available
            if self.mahalanobis.ue_bounds_tensor is not None:
                uncertainty = self.mahalanobis.normalize_ue(
                    uncertainty[0],
                    self.device
                )
            else:
                uncertainty = uncertainty[0]

            # Handle both single input and batch input
            if uncertainty.dim() == 0:  # single value
                confidence_score = 1.0 - uncertainty.cpu().item()
                return confidence_score
            else:  # batch of values
                confidence_scores = 1.0 - uncertainty.cpu().tolist()
                return confidence_scores

        except Exception as e:
            raise RuntimeError(f"Failed to calculate confidence score: {str(e)}")