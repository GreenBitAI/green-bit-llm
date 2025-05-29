import torch
import torch.nn.functional as F
from typing import Tuple


class QuantizedQwen3MoeSparseMoeBlock:
    """
    Qwen3MoeSparseMoeBlock forward method optimized for quantized models
    """

    @staticmethod
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantization-friendly MoE forward implementation using vectorized strategy
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        total_tokens = batch_size * sequence_length

        # flat
        hidden_states_flat = hidden_states.reshape(total_tokens, hidden_dim)

        # 1. Route calculation
        router_logits = self.gate(hidden_states_flat)
        if len(router_logits.shape) > 2:
            router_logits = router_logits.reshape(total_tokens, -1)

        # 2. Calculating routing weight
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 3. Select top experts and weights
        routing_weights_topk, indices_topk = torch.topk(routing_weights, self.top_k, dim=1)

        # 4. Normalized top weights
        if self.norm_topk_prob:
            routing_weights_topk /= routing_weights_topk.sum(dim=1, keepdim=True)
        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

        # 5. Pre-allocated expert output storage
        # [total_tokens, top_k, hidden_dim]
        expert_outputs = torch.zeros(total_tokens, self.top_k, hidden_dim,
                                     dtype=hidden_states.dtype, device=hidden_states.device)

        # 6. Batch processing by experts
        for expert_idx in range(self.num_experts):
            # Create expert mask [total_tokens, top_k]
            expert_mask = (indices_topk == expert_idx)

            if not expert_mask.any():
                continue

            # Find a location using current experts
            token_idx, topk_idx = torch.where(expert_mask)

            if len(token_idx) == 0:
                continue

            # Batch Processing
            expert_inputs = hidden_states_flat[token_idx]
            expert_result = self.experts[expert_idx](expert_inputs)

            # Storing Results
            expert_outputs[token_idx, topk_idx] = expert_result

        # 7. Apply weights and sum
        # Expand weight dimension: [total_tokens, top_k, 1]
        weights_expanded = routing_weights_topk.unsqueeze(-1)

        # Weighted sum: [total_tokens, hidden_dim]
        final_hidden_states = (expert_outputs * weights_expanded).sum(dim=1)

        # 8. Reshape back to original shape
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

    @staticmethod
    def forward_micro_batched(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantization-friendly MoE forward implementation using micro_batched strategy
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        total_tokens = batch_size * sequence_length

        hidden_states_flat = hidden_states.reshape(total_tokens, hidden_dim)

        # Route calculation
        router_logits = self.gate(hidden_states_flat)
        if len(router_logits.shape) > 2:
            router_logits = router_logits.reshape(total_tokens, -1)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_topk, indices_topk = torch.topk(routing_weights, self.top_k, dim=1)

        if self.norm_topk_prob:
            routing_weights_topk /= routing_weights_topk.sum(dim=1, keepdim=True)
        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Fixed micro-batch size - quantization friendly
        micro_batch_size = min(8, total_tokens)  # Small fixed batch size

        for start_idx in range(0, total_tokens, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, total_tokens)

            # Still process token by token in micro batch - maintain quantization compatibility
            for token_idx in range(start_idx, end_idx):
                token_input = hidden_states_flat[token_idx:token_idx + 1]
                token_output = torch.zeros_like(token_input)

                for expert_pos in range(self.top_k):
                    expert_idx = indices_topk[token_idx, expert_pos].item()
                    expert_weight = routing_weights_topk[token_idx, expert_pos].item()

                    if expert_weight < 1e-4:
                        continue

                    expert_output = self.experts[expert_idx](token_input)
                    token_output = token_output + expert_output * expert_weight

                final_hidden_states[token_idx] = token_output[0]

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

def apply_qwen3_moe_patch():
    """
    Apply the monkey patch of Qwen3MoeSparseMoeBlock
    """
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        # Save the original method (in case we need to restore it)
        if not hasattr(Qwen3MoeSparseMoeBlock, '_original_forward'):
            Qwen3MoeSparseMoeBlock._original_forward = Qwen3MoeSparseMoeBlock.forward

        # Replace the forward method
        Qwen3MoeSparseMoeBlock.forward = QuantizedQwen3MoeSparseMoeBlock.forward

        print("Info: Successfully applied Qwen3MoeSparseMoeBlock patch for quantized models")

    except ImportError as e:
        print(f"Error: Could not apply Qwen3MoeSparseMoeBlock patch: {e}")
        print("   This might be expected if Qwen3 models are not being used")

def restore_qwen3_moe_patch():
    """
    Restore the original Qwen3MoeSparseMoeBlock forward method
    """
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        if hasattr(Qwen3MoeSparseMoeBlock, '_original_forward'):
            Qwen3MoeSparseMoeBlock.forward = Qwen3MoeSparseMoeBlock._original_forward
            delattr(Qwen3MoeSparseMoeBlock, '_original_forward')
            print("Info: Successfully restored original Qwen3MoeSparseMoeBlock forward method")

    except ImportError:
        pass