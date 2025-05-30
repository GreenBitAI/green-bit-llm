import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import LLMResult, Generation, ChatResult, ChatGeneration
from green_bit_llm.langchain import GreenBitPipeline, ChatGreenBit


class TestChatGreenBit(unittest.TestCase):

    def setUp(self):
        self.mock_pipeline = MagicMock(spec=GreenBitPipeline)
        self.mock_pipeline.pipeline = MagicMock()
        self.mock_pipeline.pipeline.tokenizer = MagicMock()
        self.mock_pipeline.pipeline.tokenizer.apply_chat_template = MagicMock(return_value="Mocked chat template")
        self.chat_model = ChatGreenBit(llm=self.mock_pipeline)

    def test_llm_type(self):
        self.assertEqual(self.chat_model._llm_type, "greenbit-chat")

    def test_messages_to_dict(self):
        """Test message conversion to dictionary format"""
        system_message = SystemMessage(content="You are an AI assistant.")
        human_message = HumanMessage(content="Hello, AI!")
        ai_message = AIMessage(content="Hello, human!")

        messages = [system_message, human_message, ai_message]
        result = self.chat_model._messages_to_dict(messages)

        expected = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Hello, AI!"},
            {"role": "assistant", "content": "Hello, human!"}
        ]
        self.assertEqual(result, expected)

    def test_prepare_prompt(self):
        """Test prompt preparation using apply_chat_template"""
        messages = [
            SystemMessage(content="You are an AI assistant."),
            HumanMessage(content="Hello, AI!"),
            AIMessage(content="Hello, human!"),
            HumanMessage(content="How are you?")
        ]

        result = self.chat_model._prepare_prompt(messages)

        self.assertEqual(result, "Mocked chat template")
        self.mock_pipeline.pipeline.tokenizer.apply_chat_template.assert_called_once()

        # Check that the call was made with correct message format
        call_args = self.mock_pipeline.pipeline.tokenizer.apply_chat_template.call_args
        messages_arg = call_args[0][0]  # First positional argument
        expected_messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Hello, AI!"},
            {"role": "assistant", "content": "Hello, human!"},
            {"role": "user", "content": "How are you?"}
        ]
        self.assertEqual(messages_arg, expected_messages)

    def test_prepare_prompt_with_enable_thinking(self):
        """Test prompt preparation with enable_thinking parameter"""
        messages = [HumanMessage(content="Hello, AI!")]

        result = self.chat_model._prepare_prompt(messages, enable_thinking=True)

        self.assertEqual(result, "Mocked chat template")
        call_args = self.mock_pipeline.pipeline.tokenizer.apply_chat_template.call_args
        kwargs = call_args[1]  # Keyword arguments
        self.assertTrue(kwargs.get("enable_thinking"))
        self.assertTrue(kwargs.get("add_generation_prompt"))

    def test_prepare_prompt_no_tokenizer(self):
        """Test error handling when tokenizer is not available"""
        self.mock_pipeline.pipeline.tokenizer = None
        messages = [HumanMessage(content="Hello, AI!")]

        with self.assertRaises(ValueError) as context:
            self.chat_model._prepare_prompt(messages)
        self.assertIn("Tokenizer not available", str(context.exception))

    def test_prepare_prompt_no_chat_template(self):
        """Test error handling when apply_chat_template is not available"""
        del self.mock_pipeline.pipeline.tokenizer.apply_chat_template
        messages = [HumanMessage(content="Hello, AI!")]

        with self.assertRaises(ValueError) as context:
            self.chat_model._prepare_prompt(messages)
        self.assertIn("does not support apply_chat_template", str(context.exception))

    def test_create_chat_result(self):
        """Test conversion from LLM result to chat result"""
        llm_result = LLMResult(generations=[[Generation(text="Hello, human!")]])
        chat_result = self.chat_model._create_chat_result(llm_result)

        self.assertEqual(len(chat_result.generations), 1)
        self.assertIsInstance(chat_result.generations[0], ChatGeneration)
        self.assertEqual(chat_result.generations[0].message.content, "Hello, human!")

    @patch.object(ChatGreenBit, '_prepare_prompt')
    def test_generate(self, mock_prepare_prompt):
        """Test generation with mocked prompt preparation"""
        mock_prepare_prompt.return_value = "Mocked chat prompt"
        self.mock_pipeline.generate.return_value = LLMResult(generations=[[Generation(text="Generated response")]])

        messages = [HumanMessage(content="Hello, AI!")]
        result = self.chat_model.generate(messages, temperature=0.8, max_tokens=100)

        # Check that prompt was prepared correctly
        mock_prepare_prompt.assert_called_once_with(messages, temperature=0.8, max_tokens=100)

        # Check that pipeline.generate was called with correct arguments
        self.mock_pipeline.generate.assert_called_once()
        call_args = self.mock_pipeline.generate.call_args

        # Check prompts argument
        self.assertEqual(call_args[1]["prompts"], ["Mocked chat prompt"])

        # Check that result is ChatResult
        self.assertIsInstance(result, ChatResult)
        self.assertEqual(result.generations[0].message.content, "Generated response")

    @patch.object(ChatGreenBit, '_prepare_prompt')
    def test_stream(self, mock_prepare_prompt):
        """Test streaming with mocked prompt preparation"""
        mock_prepare_prompt.return_value = "Mocked chat prompt"
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.text = " human!"
        self.mock_pipeline.stream.return_value = [mock_chunk1, mock_chunk2]

        messages = [HumanMessage(content="Hello, AI!")]
        stream_result = list(self.chat_model.stream(messages, temperature=0.7))

        # Check that prompt was prepared correctly
        mock_prepare_prompt.assert_called_once_with(messages, temperature=0.7)

        # Check that pipeline.stream was called
        self.mock_pipeline.stream.assert_called_once()

        # Check stream results
        self.assertEqual(len(stream_result), 2)
        self.assertIsInstance(stream_result[0], ChatGeneration)
        self.assertEqual(stream_result[0].message.content, "Hello")
        self.assertEqual(stream_result[1].message.content, " human!")

    def test_generate_with_enable_thinking(self):
        """Test generation with enable_thinking parameter"""
        with patch.object(self.chat_model, '_prepare_prompt') as mock_prepare_prompt:
            mock_prepare_prompt.return_value = "Mocked chat prompt"
            self.mock_pipeline.generate.return_value = LLMResult(generations=[[Generation(text="Generated response")]])

            messages = [HumanMessage(content="Hello, AI!")]
            result = self.chat_model.generate(messages, enable_thinking=True)

            # Check that enable_thinking was passed to _prepare_prompt
            mock_prepare_prompt.assert_called_once_with(messages, enable_thinking=True)


if __name__ == '__main__':
    unittest.main()