import unittest
from unittest.mock import patch, MagicMock
import torch
from green_bit_llm.langchain import GreenBitPipeline
from langchain_core.outputs import LLMResult, Generation
from transformers import Pipeline

class TestGreenBitPipeline(unittest.TestCase):

    @patch('green_bit_llm.langchain.pipeline.check_engine_available')
    @patch('green_bit_llm.langchain.pipeline.load')
    @patch('green_bit_llm.langchain.pipeline.pipeline')
    def test_from_model_id(self, mock_pipeline, mock_load, mock_check_engine):
        # Setup
        mock_check_engine.return_value = True
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer, None)
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        # Test
        gb_pipeline = GreenBitPipeline.from_model_id(
            model_id="test_model",
            task="text-generation",
            device="cuda:0",
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
        mock_load.assert_called_once_with("test_model", device_map={"cuda:0"}, dtype=torch.float16)
        mock_pipeline.assert_called_once()

    def test_identifying_params(self):
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
        gb_pipeline = GreenBitPipeline(pipeline=MagicMock())
        self.assertEqual(gb_pipeline._llm_type, "greenbit_pipeline")

    @patch.object(Pipeline, '__call__')
    def test_generate(self, mock_pipeline_call):
        # Setup
        mock_pipeline = MagicMock()
        mock_pipeline.task = "text-generation"
        mock_pipeline_call.return_value = [{"generated_text": "Hello world"}]
        gb_pipeline = GreenBitPipeline(
            pipeline=mock_pipeline,
            pipeline_kwargs={"max_length": 100}
        )

        # Test
        result = gb_pipeline._generate(["Hi"], stop=None)

        # Assert
        self.assertIsInstance(result, LLMResult)
        self.assertEqual(len(result.generations), 1)
        self.assertEqual(result.generations[0][0].text, "Hello world")
        mock_pipeline_call.assert_called_once_with(["Hi"], max_length=100)

    @patch('green_bit_llm.langchain.pipeline.TextIteratorStreamer')
    @patch('green_bit_llm.langchain.pipeline.Thread')
    def test_stream(self, mock_thread, mock_streamer):
        # Setup
        mock_pipeline = MagicMock()
        mock_pipeline.tokenizer.encode.return_value = [1, 2, 3]
        mock_pipeline.device = "cuda:0"
        mock_streamer_instance = MagicMock()
        mock_streamer_instance.__iter__.return_value = iter(["Hello", " ", "world"])
        mock_streamer.return_value = mock_streamer_instance
        gb_pipeline = GreenBitPipeline(pipeline=mock_pipeline)

        # Test
        chunks = list(gb_pipeline._stream("Hi", stop=["END"]))

        # Assert
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "Hello")
        self.assertEqual(chunks[1].text, " ")
        self.assertEqual(chunks[2].text, "world")
        mock_thread.assert_called_once()
        mock_streamer.assert_called_once()

if __name__ == '__main__':
    unittest.main()