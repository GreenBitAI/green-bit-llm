import re
from typing import Optional
from green_bit_llm.inference.conversation import Conversation, SeparatorStyle

class ResponseFormatHandler:
    """Helper class to handle response format extraction for different model templates"""
    
    def __init__(self, conv_template: Optional[Conversation] = None):
        self.conv_template = conv_template

    def extract_response(self, text: str, clean_template: bool = True) -> str:
        """
        Extract and clean the model's response from generated text
        
        Args:
            text: The raw generated text
            clean_template: Whether to clean template tags and markers
                          Set True for chat model, False for pipeline
        """
        if not self.conv_template:
            return text.strip()

        if self.conv_template.sep_style == SeparatorStyle.CHATML:
            return self._handle_chatml_format(text, clean_template)
        elif self.conv_template.sep_style == SeparatorStyle.LLAMA2:
            return self._handle_llama2_format(text, clean_template)
        elif self.conv_template.sep_style == SeparatorStyle.LLAMA3:
            return self._handle_llama3_format(text, clean_template)
        elif self.conv_template.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            return self._handle_no_colon_format(text, clean_template)
        elif self.conv_template.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            return self._handle_single_colon_format(text, clean_template)
        elif self.conv_template.sep_style == SeparatorStyle.ADD_COLON_TWO:
            return self._handle_two_colon_format(text, clean_template)

        # Handle special cases
        text = self._handle_special_tokens(text, clean_template)
        return text.strip()

    def validate_llama3_format(self, prompt: str) -> bool:
        """
        Validate if the prompt follows Llama-3 format requirements.

        Args:
            prompt: The input prompt to validate

        Returns:
            bool: True if format is valid, False otherwise
        """
        try:
            # Should only have one begin_of_text
            if prompt.count("<|begin_of_text|>") != 1:
                return False

            # System message should come first if present
            if "<|start_header_id|>system<|end_header_id|>" in prompt:
                system_pos = prompt.find("<|start_header_id|>system<|end_header_id|>")
                begin_pos = prompt.find("<|begin_of_text|>")
                if not (begin_pos < system_pos and system_pos == prompt.find("<|start_header_id|>", begin_pos)):
                    return False

            # Each message section should end with eot_id before next header
            sections = prompt.split("<|start_header_id|>")[1:]  # Skip first part
            for section in sections[:-1]:  # Skip last section (assistant)
                if not section.strip().endswith("<|eot_id|>"):
                    return False

            # Verify header sequence
            expected_sequence = {
                "system": 0,
                "user": 1,
                "assistant": 2
            }

            current_pos = -1
            for header in re.finditer(r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>", prompt):
                header_type = header.group(1)
                if expected_sequence[header_type] <= current_pos:
                    return False
                current_pos = expected_sequence[header_type]

            return True
        except Exception:
            return False

    def fix_llama3_format(self, prompt: str) -> str:
        """
        Fix common issues in Llama-3 format prompts.

        Args:
            prompt: The prompt to fix

        Returns:
            str: Fixed prompt
        """
        # Remove duplicate begin_of_text tags
        if prompt.count("<|begin_of_text|>") > 1:
            # Keep only the first occurrence
            parts = prompt.split("<|begin_of_text|>")
            prompt = "<|begin_of_text|>" + "".join(parts[1:])

        # Ensure correct message order
        messages = []
        system_match = re.search(r"<\|start_header_id\|>system<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", prompt,
                                 re.DOTALL)
        user_match = re.search(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", prompt, re.DOTALL)

        # Build proper sequence
        result = "<|begin_of_text|>"
        if system_match:
            result += f"<|start_header_id|>system<|end_header_id|>\n\n{system_match.group(1)}<|eot_id|>"
        if user_match:
            result += f"<|start_header_id|>user<|end_header_id|>\n\n{user_match.group(1)}<|eot_id|>"
        result += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return result

    def _handle_chatml_format(self, text: str, clean_template: bool) -> str:
        """Handle ChatML format (Qwen, Yi-chat)"""
        if "<|im_start|>assistant:" in text:
            parts = text.split("<|im_start|>assistant:")
            if len(parts) > 1:
                response = parts[-1]
                response = response.split("<|im_start|>user:")[0]
                response = response.split("<|im_end|>")[0]
                
                if clean_template:
                    response = response.replace("<|im_start|>system:", "")
                    response = response.replace("<|im_start|>user:", "")
                    response = response.replace("<|im_start|>assistant:", "")
                    response = response.replace("<|im_end|>", "")
                return response.strip()
                
        if "assistant\n" in text:
            parts = text.split("assistant\n")
            if len(parts) > 1:
                response = parts[-1].split("<|im_end|>")[0]
                return response.strip()
                
        return text.strip()

    def _handle_llama2_format(self, text: str, clean_template: bool) -> str:
        """Handle Llama-2 format"""
        if "[/INST]" in text:
            response = text.split("[/INST]")[-1]
            response = response.split("[INST]")[0]
            response = response.split("</s>")[0]
            
            if clean_template:
                response = response.replace("<<SYS>>", "").replace("<</SYS>>", "")
            return response.strip()
            
        return text.strip()

    def _handle_llama3_format(self, text: str, clean_template: bool) -> str:
        """Handle Llama-3 format responses"""
        if "<|start_header_id|>assistant<|end_header_id|>" in text:
            # Split on assistant header
            parts = text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
            if len(parts) > 1:
                response = parts[-1]

                # Remove content after next header or eot_id
                response = response.split("<|eot_id|>")[0]
                response = response.split("<|start_header_id|>")[0]

                # Clean up template markers if requested
                if clean_template:
                    response = response.replace("<|begin_of_text|>", "")
                    response = response.replace("<|start_header_id|>", "")
                    response = response.replace("<|end_header_id|>", "")
                    response = response.replace("<|eot_id|>", "")
                    # Clean up extra newlines
                    response = re.sub(r'\n{3,}', '\n\n', response)

                return response.strip()

        return text.strip()

    def _handle_no_colon_format(self, text: str, clean_template: bool) -> str:
        """Handle No-Colon format (Gemma)"""
        if "<start_of_turn>model\n" in text:
            response = text.split("<start_of_turn>model\n")[-1]
            response = response.split("<end_of_turn>")[0]
            response = response.split("<start_of_turn>user")[0]
            return response.strip()
            
        return text.strip()

    def _handle_single_colon_format(self, text: str, clean_template: bool) -> str:
        """Handle Single-Colon format (including Mistral)"""
        if f"{self.conv_template.roles[1]}:" in text:
            response = text.split(f"{self.conv_template.roles[1]}:")[-1]
            response = response.split(f"{self.conv_template.roles[0]}:")[0]
            return response.strip()
            
        return text.strip()

    def _handle_two_colon_format(self, text: str, clean_template: bool) -> str:
        """Handle Two-Colon format"""
        if f"{self.conv_template.roles[1]}:" in text:
            response = text.split(f"{self.conv_template.roles[1]}:")[-1]
            if self.conv_template.sep2 in response:
                response = response.split(self.conv_template.sep2)[0]
            else:
                response = response.split(self.conv_template.sep)[0]
            return response.strip()
            
        return text.strip()

    def _handle_special_tokens(self, text: str, clean_template: bool) -> str:
        """Handle special tokens for various models"""
        # Handle phi-3, TinyLlama
        if "</s>" in text:
            text = text.split("</s>")[0]
            if clean_template:
                for tag in ["<|system|>", "<|user|>", "<|assistant|>"]:
                    if tag in text:
                        text = text.split(tag)[-1]

        # Handle Gemini
        if "model\n" in text:
            text = text.split("model\n")[-1]
            text = text.split("user\n")[0]

        return text
