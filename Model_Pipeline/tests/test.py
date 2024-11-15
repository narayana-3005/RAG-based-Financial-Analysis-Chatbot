# Imports
import unittest
import numpy as np

class TestRAGPipeline(unittest.TestCase):
    """
           This is test class for RAG Model pipeline functions.
    """
    def setUp(self):
        # Example documents (same as in your code)
        self.documents = [
            "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
            "The Amazon rainforest is a large tropical rainforest in South America, known for its biodiversity and vast ecosystems. It is often referred to as the 'lungs of the Earth'.",
            "The capital of Japan is Tokyo. It is known for its towers, including the Tokyo Skytree, and its busy districts like Shibuya and Shinjuku.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the tower.",
            "Tom is friend of jerry."
        ]
        self.index = build_faiss_index(self.documents)

    def test_build_faiss_index(self):
        """
           This is test method for faiss index.
        """
        # Check if the index is built correctly (e.g., it's not None and has the right dimension)
        self.assertIsNotNone(self.index)
        self.assertEqual(self.index.ntotal, len(self.documents))

    def test_retrieve_context(self):
        """
           This is test method for RAG Model pipeline retrieve_context function.
        """
        query = "What is capital of Japan?"
        retrieved_docs = retrieve_context(query, self.index, self.documents)
        self.assertEqual(len(retrieved_docs), 3)  # Check if 3 documents are retrieved (k=3 by default)
        self.assertIn("The capital of Japan is Tokyo.", " ".join(retrieved_docs))  # Check if relevant document is in the result

    def test_generate_answer(self):
        """
           This is test method for RAG Model pipeline retrieve_context and generate_answer functions.
        """
        query = "Who is friend of Tom?"
        retrieved_docs = retrieve_context(query, self.index, self.documents)
        answer = generate_answer(query, retrieved_docs)
        self.assertIn("jerry", answer)

        query = "What is capital of Japan?"
        retrieved_docs = retrieve_context(query, self.index, self.documents)
        answer = generate_answer(query, retrieved_docs)
        self.assertIn("tokyo", answer)  # Check if the answer contains the expected keyword.

        query = "Who designed the Eiffel Tower?"
        retrieved_docs = retrieve_context(query, self.index, self.documents)
        answer = generate_answer(query, retrieved_docs)
        self.assertIn("gustave eiffel", answer)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
