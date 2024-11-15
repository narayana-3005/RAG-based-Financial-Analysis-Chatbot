# !pip install transformers datasets faiss-cpu google-cloud-storage

# Imports
import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRReader, DPRReaderTokenizer
import faiss
import numpy as np

# Initialize Tokenizer and Models for the encoder (Question Encoder) and generator (Reader for QA)
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
reader_model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")

# FAISS Index Setup
def build_faiss_index(documents):
    """Build FAISS index for the context documents."""
    context_embeddings = []
    for doc in documents:
        # Tokenize and process each document
        inputs = reader_tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = context_encoder(**inputs).pooler_output
        context_embeddings.append(embeddings.cpu().numpy())

    # Convert the embeddings list to a numpy array
    context_embeddings = np.vstack(context_embeddings)

    # FAISS index for efficient retrieval
    index = faiss.IndexFlatL2(context_embeddings.shape[1])  # Flat L2 index
    index.add(context_embeddings)
    return index

# Function to retrieve the top-k relevant documents based on the query
def retrieve_context(query, index, documents, k=3):
    """Retrieve top-k documents relevant to the query."""
    # Encode the query with the question encoder
    inputs = reader_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        query_embedding = question_encoder(**inputs).pooler_output.cpu().numpy()

    # Perform the search on the FAISS index
    D, I = index.search(query_embedding, k)  # D = distances, I = indices of retrieved documents

    # Check if indices are valid and retrieve documents
    retrieved_docs = []
    for i in I[0]:
        if i < len(documents):  # Check to avoid out-of-bounds access
            retrieved_docs.append(documents[i])
        else:
            print(f"Warning: Retrieved index {i} is out of range for documents list.")
    return retrieved_docs

# Function to generate the answer using the DPRReader (Facebook's model for QA)
def generate_answer(query, retrieved_docs):
    """Generate an answer to the query using the DPRReader model."""
    # Combine retrieved documents into a context for the model
    context = " ".join(retrieved_docs)
    print("Context: ", context)

    # Encode the query and context to generate the answer
    inputs = reader_tokenizer(query, context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = reader_model(**inputs)

    # Get the best answer (usually in the 'start' and 'end' logits)
    start_idx = outputs.start_logits.argmax()
    end_idx = outputs.end_logits.argmax()

    # Decode the answer from the tokens
    answer_tokens = inputs.input_ids[0][start_idx:end_idx + 1]
    answer = reader_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

# Example documents
documents = [
    "Apple's stock price has seen significant fluctuations recently. The company reached a high of $300 per share before dipping, driven by challenges in the global tech market and supply chain disruptions.",
    "Tesla's stock has reached new highs, now trading at $900 per share. Strong earnings reports and growing optimism about the electric vehicle market have driven the increase, with investors expecting further growth in the coming year.",
    "Amazon's stock saw a rise to $150 in March 2024, but concerns about market volatility have led to mixed predictions. Some analysts believe Amazon will continue to perform well, while others are wary of potential downturns in the retail sector.",
    "The Federal Reserve raised interest rates by 0.25% in early 2024, causing increased volatility in the tech sector. Companies like Apple, Tesla, and Amazon have felt the effects, with fluctuations in their stock prices reflecting investor reactions to the rate hikes.",
    "Microsoft stock price has surged following strong performance in its cloud business, reaching $380 per share. Analysts are optimistic about the company's growth as more businesses adopt cloud-based solutions.",
    "Meta stock price has dropped 25 % after announcing major layoffs, but the company remains committed to investing in the metaverse. Investors are divided on the long-term profitability of Meta's new focus."
]
queries = [
    "What is the recent trend in Apple's stock price?",
    "How did Tesla's stock perform after the latest earnings report?",
    "How much is federal reserve interest rate in early 2024?"
]


# Build FAISS index from the documents
index = build_faiss_index(documents)

for query in queries:

    # Get the retrieved context documents
    retrieved_docs = retrieve_context(query, index, documents)

    # Generate the answer using the DPRReader model
    answer = generate_answer(query, retrieved_docs)
    print('Query: ', query)
    print("Answer:", answer)



