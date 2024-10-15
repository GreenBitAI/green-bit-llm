import unittest
from typing import List
from green_bit_llm.langchain import GreenBitEmbeddings

class TestGreenBitEmbeddings(unittest.TestCase):
    def setUp(self):
        model_kwargs = {'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = GreenBitEmbeddings.from_model_id(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="cache",
            device="cpu",
            multi_process=False,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def test_embed_documents_returns_list(self):
        texts = ["Hello, world!", "This is a test."]
        result = self.embeddings.embed_documents(texts)
        self.assertIsInstance(result, list)

    def test_embed_documents_returns_correct_number_of_embeddings(self):
        texts = ["Hello, world!", "This is a test."]
        result = self.embeddings.embed_documents(texts)
        self.assertEqual(len(result), len(texts))

    def test_embed_query_returns_list(self):
        query = "What is the meaning of life?"
        result = self.embeddings.embed_query(query)
        self.assertIsInstance(result, list)

    def test_embed_query_returns_non_empty_list(self):
        query = "What is the meaning of life?"
        result = self.embeddings.embed_query(query)
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    unittest.main()