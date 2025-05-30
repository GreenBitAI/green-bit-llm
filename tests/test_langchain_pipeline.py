import unittest
from unittest.mock import patch, MagicMock
import torch
from green_bit_llm.langchain import GreenBitPipeline
from langchain_core.outputs import LLMResult, Generation, GenerationChunk


class TestGreenBitPipeline(unittest.TestCase):

    @patch('green_bit_llm.langchain.pipeline.check_engine_available')
    @patch('green_bit_llm.langchain.pipeline.load')
    @patch('green_bit_llm.langchain.pipeline.pipeline')
    def test_from_model_id(self, mock_pipeline, mock_load, mock_check_engine):
        """Test pipeline creation from model ID"""
        # Setup
        mock_check_engine.return_value = True
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 50256
        mock_load.return_value = (mock_model, mock_tokenizer, None)
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        # Test
        gb_pipeline = GreenBitPipeline.from_model_id(
            model_id="test_model",
            task="text-generation",
            model_kwargs={"dtype": torch.float16},
            pipeline_kwargs={"max_length": 100}
        )

        # Assert
        self.assertIsInstance(gb_pipeline, GreenBitPipeline)
        self.assertEqual(gb_pipeline.model_id, "test_model")
        self.assertEqual(gb_pipeline.task, "text-generation")
        self.assertEqual(gb_pipeline.model_kwargs, {"dtype": torch.float16})
        self.assertEqual(gb_pipeline.pipeline_kwargs, {"max_length": 100})
        mock_check_engine.assert_called_once()
        mock_pipeline.assert_called_once()

        # Check that tokenizer pad_token was set
        self.assertEqual(mock_tokenizer.pad_token, "<|endoftext|>")
        self.assertEqual(mock_tokenizer.pad_token_id, 50256)

    def test_identifying_params(self):
        """Test identifying parameters property"""
        gb_pipeline = GreenBitPipeline(
            pipeline=MagicMock(),
            model_id="test_model",
            task="text-generation",
            model_kwargs={"dtype": torch.float16},
            pipeline_kwargs={"max_length": 100}
        )
        params = gb_pipeline._identifying_params
        self.assertEqual(params["model_id"], "test_model")
        self.assertEqual(params["task"], "text-generation")
        self.assertEqual(params["model_kwargs"], {"dtype": torch.float16})
        self.assertEqual(params["pipeline_kwargs"], {"max_length": 100})

    def test_llm_type(self):
        """Test LLM type property"""
        gb_pipeline = GreenBitPipeline(pipeline=MagicMock(), model_kwargs={}, pipeline_kwargs={})
        self.assertEqual(gb_pipeline._llm_type, "greenbit_pipeline")

    def test_prepare_generation_config(self):
        """Test generation config preparation"""
        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer.pad_token_id = 0
        mock_pipeline.tokenizer.eos_token_id = 1

        gb_pipeline = GreenBitPipeline(
            pipeline=mock_pipeline,
            model_kwargs={},
            pipeline_kwargs={"temperature": 0.5, "max_new_tokens": 50}
        )

        config = gb_pipeline._prepare_generation_config({"temperature": 0.8})

        self.assertEqual(config["temperature"], 0.8)  # Should override pipeline_kwargs
        self.assertEqual(config["max_new_tokens"], 50)  # Should use pipeline default
        self.assertEqual(config["pad_token_id"], 0)
        self.assertEqual(config["eos_token_id"], 1)
        self.assertTrue(config["do_sample"])  # Changed default to True

    def test_prepare_prompt_from_text_with_chat_template(self):
        """Test prompt preparation from plain text using chat template"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(
            return_value="<|im_start|>user\nHello<|im_end|><|im_start|>assistant\n")

        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer = mock_tokenizer

        gb_pipeline = GreenBitPipeline(pipeline=mock_pipeline, model_kwargs={}, pipeline_kwargs={})

        result = gb_pipeline._prepare_prompt_from_text("Hello", enable_thinking=True)

        self.assertEqual(result, "<|im_start|>user\nHello<|im_end|><|im_start|>assistant\n")
        mock_tokenizer.apply_chat_template.assert_called_once()

        # Check the call arguments
        call_args = mock_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        kwargs = call_args[1]

        self.assertEqual(messages, [{"role": "user", "content": "Hello"}])
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertTrue(kwargs["enable_thinking"])

    def test_prepare_prompt_from_text_no_chat_template(self):
        """Test prompt preparation fallback when no chat template available"""
        mock_tokenizer = MagicMock()
        # Remove apply_chat_template method
        del mock_tokenizer.apply_chat_template

        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer = mock_tokenizer

        gb_pipeline = GreenBitPipeline(pipeline=mock_pipeline, model_kwargs={}, pipeline_kwargs={})

        result = gb_pipeline._prepare_prompt_from_text("Hello")

        # Should return original text when no chat template
        self.assertEqual(result, "Hello")

    def test_prepare_prompt_from_text_template_error(self):
        """Test prompt preparation fallback when template application fails"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(side_effect=Exception("Template error"))

        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer = mock_tokenizer

        gb_pipeline = GreenBitPipeline(pipeline=mock_pipeline, model_kwargs={}, pipeline_kwargs={})

        result = gb_pipeline._prepare_prompt_from_text("Hello")

        # Should return original text when template application fails
        self.assertEqual(result, "Hello")

    @patch.object(GreenBitPipeline, '_prepare_prompt_from_text')
    def test_generate_with_plain_text(self, mock_prepare_prompt):
        """Test generation with plain text prompts"""
        # Setup
        mock_prepare_prompt.return_value = "<formatted_prompt>"
        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer.encode.return_value = [1, 2, 3]
        mock_pipeline.device = "cpu"
        mock_pipeline.model.generate.return_value = MagicMock(
            sequences=torch.tensor([[1, 2, 3, 4, 5]]),
            hidden_states=None
        )
        mock_pipeline.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_pipeline.tokenizer.decode.return_value = "Generated text"

        gb_pipeline = GreenBitPipeline(
            pipeline=mock_pipeline,
            model_kwargs={},
            pipeline_kwargs={"max_new_tokens": 100}
        )

        # Test with plain text (should trigger chat template)
        result = gb_pipeline.generate(["Hello"], enable_thinking=True)

        # Check that prompt was processed
        mock_prepare_prompt.assert_called_once_with("Hello", enable_thinking=True)

        # Check result
        self.assertIsInstance(result, LLMResult)
        self.assertEqual(len(result.generations), 1)

    def test_generate_with_formatted_prompt(self):
        """Test generation with already formatted prompts"""
        # Setup
        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer.encode.return_value = [1, 2, 3]
        mock_pipeline.device = "cpu"
        mock_pipeline.model.generate.return_value = MagicMock(
            sequences=torch.tensor([[1, 2, 3, 4, 5]]),
            hidden_states=None
        )
        mock_pipeline.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_pipeline.tokenizer.decode.return_value = "Generated text"

        gb_pipeline = GreenBitPipeline(
            pipeline=mock_pipeline,
            model_kwargs={},
            pipeline_kwargs={"max_new_tokens": 100}
        )

        # Test with already formatted prompt (should not trigger chat template)
        formatted_prompt = "<|im_start|>user\nHello<|im_end|><|im_start|>assistant\n"

        with patch.object(gb_pipeline, '_prepare_prompt_from_text') as mock_prepare:
            result = gb_pipeline.generate([formatted_prompt])

            # Should not call _prepare_prompt_from_text for already formatted prompts
            mock_prepare.assert_not_called()

    @patch('green_bit_llm.langchain.pipeline.TextIteratorStreamer')
    @patch('green_bit_llm.langchain.pipeline.Thread')
    @patch.object(GreenBitPipeline, '_prepare_prompt_from_text')
    def test_stream(self, mock_prepare_prompt, mock_thread, mock_streamer):
        """Test streaming functionality"""
        # Setup
        mock_prepare_prompt.return_value = "<formatted_prompt>"
        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
        mock_pipeline.device = "cpu"

        mock_streamer_instance = MagicMock()
        mock_streamer_instance.__iter__.return_value = iter(["Hello", " ", "world"])
        mock_streamer.return_value = mock_streamer_instance

        gb_pipeline = GreenBitPipeline(pipeline=mock_pipeline, model_kwargs={}, pipeline_kwargs={})

        # Test
        chunks = list(gb_pipeline.stream("Hi", enable_thinking=True))

        # Check that prompt was processed
        mock_prepare_prompt.assert_called_once_with("Hi", enable_thinking=True)

        # Assert
        self.assertEqual(len(chunks), 3)
        self.assertIsInstance(chunks[0], GenerationChunk)
        self.assertEqual(chunks[0].text, "Hello")
        self.assertEqual(chunks[1].text, " ")
        self.assertEqual(chunks[2].text, "world")
        mock_thread.assert_called_once()
        mock_streamer.assert_called_once()

    @patch.object(GreenBitPipeline, '_prepare_prompt_from_text')
    def test_stream_with_formatted_prompt(self, mock_prepare_prompt):
        """Test streaming with already formatted prompt"""
        # Setup
        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
        mock_pipeline.device = "cpu"

        gb_pipeline = GreenBitPipeline(pipeline=mock_pipeline, model_kwargs={}, pipeline_kwargs={})

        formatted_prompt = "<|start_header_id|>user<|end_header_id|>Hello<|eot_id|>"

        with patch('green_bit_llm.langchain.pipeline.TextIteratorStreamer') as mock_streamer:
            with patch('green_bit_llm.langchain.pipeline.Thread'):
                mock_streamer_instance = MagicMock()
                mock_streamer_instance.__iter__.return_value = iter(["Hello"])
                mock_streamer.return_value = mock_streamer_instance

                list(gb_pipeline.stream(formatted_prompt))

        # Should not call _prepare_prompt_from_text for already formatted prompts
        mock_prepare_prompt.assert_not_called()


if __name__ == '__main__':
    unittest.main()