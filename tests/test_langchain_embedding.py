import unittest
from unittest.mock import patch, MagicMock
import torch
from green_bit_llm.langchain.embedding import GreenBitEmbeddings, DEFAULT_MODEL_NAME2

class TestGreenBitEmbeddings(unittest.TestCase):

    @patch('green_bit_llm.langchain.embedding.sentence_transformers.SentenceTransformer')
    def test_initialization(self, mock_sentence_transformer):
        embedder = GreenBitEmbeddings(model_name="test_model", device="cuda")
        mock_sentence_transformer.assert_called_once_with(
            "test_model", device="cuda", cache_folder=None
        )

    @patch('green_bit_llm.langchain.embedding.sentence_transformers.SentenceTransformer')
    def test_embed_documents(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mock_sentence_transformer.return_value = mock_model

        embedder = GreenBitEmbeddings()
        result = embedder.embed_documents(["text1", "text2"])

        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        mock_model.encode.assert_called_once_with(
            ["text1", "text2"], show_progress_bar=False
        )

    @patch('green_bit_llm.langchain.embedding.sentence_transformers.SentenceTransformer')
    def test_embed_query(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([[1.0, 2.0]])
        mock_sentence_transformer.return_value = mock_model

        embedder = GreenBitEmbeddings()
        result = embedder.embed_query("query text")

        self.assertEqual(result, [1.0, 2.0])
        mock_model.encode.assert_called_once_with(
            ["query text"], show_progress_bar=False
        )

    @patch('green_bit_llm.langchain.embedding.sentence_transformers.SentenceTransformer')
    def test_from_model_id(self, mock_sentence_transformer):
        embedder = GreenBitEmbeddings.from_model_id(
            "test_model",
            device="cuda",
            cache_dir="/tmp",
            multi_process=True,
            show_progress=True,
            model_kwargs={"test_arg": "value"},
            encode_kwargs={"batch_size": 32}
        )

        self.assertEqual(embedder.model_name, "test_model")
        self.assertEqual(embedder.device, "cuda")
        self.assertEqual(embedder.cache_dir, "/tmp")
        self.assertTrue(embedder.multi_process)
        self.assertTrue(embedder.show_progress)
        self.assertEqual(embedder.model_kwargs, {"test_arg": "value"})
        self.assertEqual(embedder.encode_kwargs, {"batch_size": 32})

    def test_default_values(self):
        embedder = GreenBitEmbeddings()
        self.assertEqual(embedder.model_name, DEFAULT_MODEL_NAME2)
        self.assertEqual(embedder.device, "cuda")
        self.assertFalse(embedder.multi_process)
        self.assertFalse(embedder.show_progress)

    @patch('green_bit_llm.langchain.embedding.sentence_transformers')
    def test_multi_process(self, mock_sentence_transformers):
        mock_model = MagicMock()
        mock_pool = MagicMock()
        mock_model.start_multi_process_pool.return_value = mock_pool
        mock_model.encode_multi_process.return_value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        embedder = GreenBitEmbeddings(multi_process=True)
        result = embedder.embed_documents(["text1", "text2"])

        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        mock_model.start_multi_process_pool.assert_called_once()
        mock_model.encode_multi_process.assert_called_once_with(["text1", "text2"], mock_pool)
        mock_sentence_transformers.SentenceTransformer.stop_multi_process_pool.assert_called_once_with(mock_pool)

if __name__ == '__main__':
    unittest.main()