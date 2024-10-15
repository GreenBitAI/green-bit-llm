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

    def test_to_chatml_format(self):
        system_message = SystemMessage(content="You are an AI assistant.")
        human_message = HumanMessage(content="Hello, AI!")
        ai_message = AIMessage(content="Hello, human!")

        self.assertEqual(self.chat_model._to_chatml_format(system_message), {"role": "system", "content": "You are an AI assistant."})
        self.assertEqual(self.chat_model._to_chatml_format(human_message), {"role": "user", "content": "Hello, AI!"})
        self.assertEqual(self.chat_model._to_chatml_format(ai_message), {"role": "assistant", "content": "Hello, human!"})

        with self.assertRaises(ValueError):
            self.chat_model._to_chatml_format(MagicMock())

    def test_to_chat_prompt(self):
        messages = [
            SystemMessage(content="You are an AI assistant."),
            HumanMessage(content="Hello, AI!"),
            AIMessage(content="Hello, human!"),
            HumanMessage(content="How are you?")
        ]

        result = self.chat_model._to_chat_prompt(messages)

        self.assertEqual(result, "Mocked chat template")
        self.mock_pipeline.pipeline.tokenizer.apply_chat_template.assert_called_once()

    def test_to_chat_result(self):
        llm_result = LLMResult(generations=[[Generation(text="Hello, human!")]])
        chat_result = self.chat_model._to_chat_result(llm_result)

        self.assertEqual(len(chat_result.generations), 1)
        self.assertIsInstance(chat_result.generations[0], ChatGeneration)
        self.assertEqual(chat_result.generations[0].message.content, "Hello, human!")

    @patch('green_bit_llm.langchain.chat_model.ChatGreenBit._to_chat_prompt')
    def test_generate(self, mock_to_chat_prompt):
        mock_to_chat_prompt.return_value = "Mocked chat prompt"
        self.mock_pipeline._generate.return_value = LLMResult(generations=[[Generation(text="Generated response")]])

        messages = [HumanMessage(content="Hello, AI!")]
        result = self.chat_model._generate(messages)

        self.mock_pipeline._generate.assert_called_once_with(prompts=["Mocked chat prompt"], stop=None, run_manager=None)
        self.assertIsInstance(result, ChatResult)
        self.assertEqual(result.generations[0].message.content, "Generated response")

    def test_stream(self):
        messages = [HumanMessage(content="Hello, AI!")]
        self.mock_pipeline._stream.return_value = [MagicMock(text="Hello"), MagicMock(text=" human!")]

        stream_result = list(self.chat_model.stream(messages))

        self.assertEqual(len(stream_result), 2)
        self.assertIsInstance(stream_result[0], ChatGeneration)
        self.assertEqual(stream_result[0].message.content, "Hello")
        self.assertEqual(stream_result[1].message.content, " human!")

if __name__ == '__main__':
    unittest.main()