# !pip install transformers datasets faiss-cpu google-cloud-storage

# Imports
import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRReader, DPRReaderTokenizer
import faiss
import numpy as np
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key and Google Cloud credentials from environment variables
SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BUCKET_NAME_DATA = os.getenv('PREPROCESSED_DATA') # "bucket_preprocessed"
BUCKET_NAME_QUERIES = os.getenv('QUERIES')

# Initialize Tokenizer and Models for the encoder (Question Encoder) and generator (Reader for QA)
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
reader_model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")


def list_and_download_parquet_blobs(bucket_name):
    """Lists all the Parquet blobs in the specified bucket and returns their content as a list of raw text."""
    logger.info("Method: list_and_download_parquet_blobs execution started.")
    # List to hold raw content of the Parquet files
    parquet_data = []

    # Initialize the Cloud Storage client
    storage_client = storage.Client()

    try:
        # Retrieve all blobs (objects) in the specified bucket
        blobs = storage_client.list_blobs(bucket_name)

        # Iterate through each blob in the bucket
        for blob in blobs:
            # Check if the blob is a Parquet file
            if blob.name.endswith('.parquet'):
                try:
                    # Download the blob's content as a byte stream
                    parquet_bytes = blob.download_as_string()

                    # Read the Parquet data using pyarrow
                    buffer = io.BytesIO(parquet_bytes)
                    table = pq.read_table(buffer)

                    # Convert the Parquet table to a pandas DataFrame (optional step)
                    df = table.to_pandas()

                    # Append the DataFrame (or just the raw Parquet data) to the list
                    parquet_data= parquet_data + (df.summary.to_list())

                except Exception as e:
                    logger.error(f"Failed to process {blob.name}: {e}")
    except Exception as e:
        logger.error(f"Error accessing bucket '{bucket_name}': {e}")
    logger.info("Method: list_and_download_parquet_blobs execution completed.")
    # Return the list of DataFrames or raw Parquet data
    return parquet_data

# FAISS Index Setup
def build_faiss_index(documents):
    """Build FAISS index for the context documents."""
    logger.info("Method: build_faiss_index execution started.")
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
    logger.info("Method: build_faiss_index execution completed.")
    return index

# Function to retrieve the top-k relevant documents based on the query
def retrieve_context(query, index, documents, k=3):
    """Retrieve top-k documents relevant to the query."""
    logger.info("Method: retrieve_context execution started.")
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
            logger.info(f"Warning: Retrieved index {i} is out of range for documents list.")
    logger.info("Method: retrieve_context execution completed.")
    return retrieved_docs

# Function to generate the answer using the DPRReader (Facebook's model for QA)
def generate_answer(query, retrieved_docs):
    """Generate an answer to the query using the DPRReader model."""
    logger.info("Method: generate_answer execution started.")
    # Combine retrieved documents into a context for the model
    context = " ".join(retrieved_docs)
    # print("Context: ", context)

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
    logger.info("Method: generate_answer execution completed.")
    return answer

# Example documents and queries
# documents_data = [
#     "Apple's stock price has seen significant fluctuations recently. The company reached a high of $300 per share before dipping, driven by challenges in the global tech market and supply chain disruptions.",
#     "Tesla's stock has reached new highs, now trading at $900 per share. Strong earnings reports and growing optimism about the electric vehicle market have driven the increase, with investors expecting further growth in the coming year.",
#     "Amazon's stock saw a rise to $150 in March 2024, but concerns about market volatility have led to mixed predictions. Some analysts believe Amazon will continue to perform well, while others are wary of potential downturns in the retail sector.",
#     "The Federal Reserve raised interest rates by 0.25% in early 2024, causing increased volatility in the tech sector. Companies like Apple, Tesla, and Amazon have felt the effects, with fluctuations in their stock prices reflecting investor reactions to the rate hikes.",
#     "Microsoft stock price has surged following strong performance in its cloud business, reaching $380 per share. Analysts are optimistic about the company's growth as more businesses adopt cloud-based solutions.",
#     "Meta stock price has dropped 25 % after announcing major layoffs, but the company remains committed to investing in the metaverse. Investors are divided on the long-term profitability of Meta's new focus."
# ]
# queries = [
#     "What is the recent trend in Apple's stock price?",
#     "How did Tesla's stock perform after the latest earnings report?",
#     "How much is federal reserve interest rate in early 2024?"
# ]

# Load Market data
documents_data = list_and_download_parquet_blobs(BUCKET_NAME_DATA)

# Load Queries
queries = list_and_download_parquet_blobs(BUCKET_NAME_QUERIES)

# Build FAISS index from the documents
index = build_faiss_index(documents)

# Example of RAG Chatbot Pipeline execution
for query in queries:

    # Get the retrieved context documents
    retrieved_docs = retrieve_context(query, index, documents)

    # Generate the answer using the DPRReader model
    answer = generate_answer(query, retrieved_docs)
    print('Query: ', query)
    print("Answer:", answer)

import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRReader, DPRReaderTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
# ... (rest of your existing code for initialization, FAISS, etc.)

# Example training data (small dataset)
training_data = [
    {"question": "What is the capital of France?", "context": "The capital of France is Paris.", "answer": "Paris"},
    {"question": "Who designed the Eiffel Tower?", "context": "The Eiffel Tower was named after the engineer Gustave Eiffel, whose company designed and built the tower.", "answer": "Gustave Eiffel"},
    {"question": "What is Python?", "context": "Python is a high-level, interpreted programming language.", "answer": "a high-level, interpreted programming language"},
    {"question": "Where is the Amazon rainforest?", "context": "The Amazon rainforest is a large tropical rainforest in South America.", "answer": "South America"}
]

# Example training data (small dataset)
eval_data = [
    {"question": "What is the capital of France?", "context": "The capital of France is Paris.", "answer": "Paris"},
    {"question": "Who designed the Eiffel Tower?", "context": "The Eiffel Tower was named after the engineer Gustave Eiffel, whose company designed and built the tower.", "answer": "Gustave Eiffel"},
]
# Convert the training data to a Hugging Face Dataset
train_dataset = Dataset.from_list(training_data)
eval_dataset = Dataset.from_list(eval_data)

# Function to prepare training data
def prepare_train_features(examples):
    tokenized_examples = reader_tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    # Add start and end positions of the answer
    start_positions = []
    end_positions = []
    for i in range(len(examples['question'])):
      answer_start = examples['context'][i].find(examples['answer'][i])
      answer_end = answer_start + len(examples['answer'][i])
      start_positions.append(answer_start)
      end_positions.append(answer_end)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

# Prepare the training dataset
train_dataset = train_dataset.map(prepare_train_features, batched=True)
eval_dataset = eval_dataset.map(prepare_train_features, batched=True)









import torch
from transformers import DPRReader, DPRReaderTokenizerFast, AdamW
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Initialize the model and tokenizer
reader_model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
reader_tokenizer = DPRReaderTokenizerFast.from_pretrained("facebook/dpr-reader-single-nq-base")

# Example training and evaluation data
training_data = [
    {"question": "What is the capital of France?", "context": "The capital of France is Paris.", "answer": "Paris"},
    {"question": "Who designed the Eiffel Tower?", "context": "The Eiffel Tower was named after the engineer Gustave Eiffel, whose company designed and built the tower.", "answer": "Gustave Eiffel"},
]

eval_data = [
    {"question": "What is the capital of France?", "context": "The capital of France is Paris.", "answer": "Paris"},
    {"question": "Who designed the Eiffel Tower?", "context": "The Eiffel Tower was named after the engineer Gustave Eiffel, whose company designed and built the tower.", "answer": "Gustave Eiffel"},
]

# Prepare training features
def prepare_train_features(example):
    tokenized_example = reader_tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    answer_start = example['context'].find(example['answer'])
    answer_end = answer_start + len(example['answer'])

    start_positions = tokenized_example.char_to_token(0, answer_start)
    end_positions = tokenized_example.char_to_token(0, answer_end - 1)

    if start_positions is None or end_positions is None:
        start_positions = 0
        end_positions = 0

    tokenized_example["start_positions"] = start_positions
    tokenized_example["end_positions"] = end_positions
    return tokenized_example

# Convert data to Dataset and map features
train_dataset = Dataset.from_list(training_data)
eval_dataset = Dataset.from_list(eval_data)

train_dataset = train_dataset.map(prepare_train_features, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(prepare_train_features, remove_columns=eval_dataset.column_names)

# Custom data collator to convert batch items to tensors
def custom_collate_fn(batch):
    return {key: torch.tensor([item[key] for item in batch]) for key in batch[0]}

# Define DataLoader with custom collator
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=2, collate_fn=custom_collate_fn)

# Set up optimizer
optimizer = AdamW(reader_model.parameters(), lr=5e-5)

# Training Loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reader_model.to(device)

for epoch in range(num_epochs):
    reader_model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # Forward pass (without start_positions and end_positions)
        outputs = reader_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Compute loss manually
        start_loss = torch.nn.functional.cross_entropy(start_logits, start_positions)
        end_loss = torch.nn.functional.cross_entropy(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        total_loss += loss.item()

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss}")

# Save the fine-tuned model
reader_model.save_pretrained("./my_fine_tuned_dpr_reader")
reader_tokenizer.save_pretrained("./my_fine_tuned_dpr_reader")

import torch
from sklearn.metrics import accuracy_score
from transformers import DPRReader, DPRReaderTokenizerFast
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assume the model is already fine-tuned and saved
# Load the fine-tuned model and tokenizer
reader_model = DPRReader.from_pretrained("./my_fine_tuned_dpr_reader")
reader_tokenizer = DPRReaderTokenizerFast.from_pretrained("./my_fine_tuned_dpr_reader")

# Validation data - Ensure it includes questions, contexts, and expected answers
validation_data = [
    {"question": "What is the capital of France?", "context": "The capital of France is Paris.", "answer": "Paris"},
    {"question": "Who designed the Eiffel Tower?", "context": "The Eiffel Tower was named after the engineer Gustave Eiffel, whose company designed and built the tower.", "answer": "Gustave Eiffel"},
]

# Prepare validation features
def prepare_validation_features(example):
    tokenized_example = reader_tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    answer_start = example['context'].find(example['answer'])
    answer_end = answer_start + len(example['answer'])

    start_positions = tokenized_example.char_to_token(0, answer_start)
    end_positions = tokenized_example.char_to_token(0, answer_end - 1)

    if start_positions is None or end_positions is None:
        start_positions = 0
        end_positions = 0

    tokenized_example["start_positions"] = start_positions
    tokenized_example["end_positions"] = end_positions
    return tokenized_example

# Convert validation data to Dataset and map features
val_dataset = Dataset.from_list(validation_data)
val_dataset = val_dataset.map(prepare_validation_features, remove_columns=val_dataset.column_names)

# Custom data collator to convert batch items to tensors
def custom_collate_fn(batch):
    return {key: torch.tensor([item[key] for item in batch]) for key in batch[0]}

# Define DataLoader with custom collator
val_dataloader = DataLoader(val_dataset, batch_size=2, collate_fn=custom_collate_fn)

# Validation Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reader_model.to(device)
reader_model.eval()

total_val_loss = 0
all_start_preds = []
all_end_preds = []
all_start_labels = []
all_end_labels = []

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="Validation"):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # Forward pass to get logits
        outputs = reader_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Compute validation loss
        start_loss = torch.nn.functional.cross_entropy(start_logits, start_positions)
        end_loss = torch.nn.functional.cross_entropy(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        total_val_loss += loss.item()

        # Calculate predictions
        start_preds = torch.argmax(start_logits, dim=1)
        end_preds = torch.argmax(end_logits, dim=1)

        # Store predictions and actual labels
        all_start_preds.extend(start_preds.cpu().tolist())
        all_end_preds.extend(end_preds.cpu().tolist())
        all_start_labels.extend(start_positions.cpu().tolist())
        all_end_labels.extend(end_positions.cpu().tolist())

# Calculate average validation loss
avg_val_loss = total_val_loss / len(val_dataloader)
print(f"Validation Loss: {avg_val_loss}")

# Calculate accuracy for start and end positions
start_accuracy = accuracy_score(all_start_labels, all_start_preds)
end_accuracy = accuracy_score(all_end_labels, all_end_preds)
print(f"Start Position Accuracy: {start_accuracy * 100:.2f}%")
print(f"End Position Accuracy: {end_accuracy * 100:.2f}%")


from transformers import pipeline

# Load a sentiment analysis pipeline from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis")

# Define keywords or phrases that could indicate potential biases
bias_keywords = ["high", "low", "surge", "drop", "positive", "negative", "crash", "volatile", "disruptions", "optimistic", "pessimistic"]

# Function to detect bias in retrieved contexts
def detect_bias_in_context(retrieved_docs):
    bias_detected = False
    bias_details = {
        "biased_documents": [],
        "sentiment_analysis": [],
        "keyword_hits": []
    }

    for doc in retrieved_docs:
        keyword_hits = [keyword for keyword in bias_keywords if keyword.lower() in doc.lower()]
        if keyword_hits:
            bias_detected = True
            bias_details["keyword_hits"].append({"document": doc, "keywords": keyword_hits})
        sentiment = sentiment_analyzer(doc)[0]
        bias_details["sentiment_analysis"].append({"document": doc, "sentiment": sentiment})
        if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.75 or \
           sentiment["label"] == "POSITIVE" and sentiment["score"] > 0.75:
            bias_detected = True
            bias_details["biased_documents"].append({"document": doc, "sentiment": sentiment})

    return bias_detected, bias_details

# Function to detect bias in generated answer
def detect_bias_in_answer(answer):
    keyword_hits = [keyword for keyword in bias_keywords if keyword.lower() in answer.lower()]
    sentiment = sentiment_analyzer(answer)[0]
    bias_detected = bool(keyword_hits) or \
                    (sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.75) or \
                    (sentiment["label"] == "POSITIVE" and sentiment["score"] > 0.75)

    bias_details = {
        "sentiment": sentiment,
        "keyword_hits": keyword_hits
    }

    return bias_detected, bias_details

# Mock functions for context retrieval and answer generation
def retrieve_context(query, index, documents):
    return ["This is a mock document relevant to the query."]

def generate_answer(query, retrieved_docs):
    return f"Mock answer for query: {query}"

# Example queries
queries = [
    "What is the impact of climate change on agriculture?",
    "What are the latest advancements in renewable energy?",
    "Explain the stock price trend for Apple."
]

# Run the pipeline
for query in queries:
    retrieved_docs = retrieve_context(query, None, None)
    context_bias_detected, context_bias_details = detect_bias_in_context(retrieved_docs)
    answer = generate_answer(query, retrieved_docs)
    answer_bias_detected, answer_bias_details = detect_bias_in_answer(answer)

    print("Query:", query)
    print("Answer:", answer)
    print("Context Bias Detected:", context_bias_detected)
    if context_bias_detected:
        print("Context Bias Details:", context_bias_details)
    print("Answer Bias Detected:", answer_bias_detected)
    if answer_bias_detected:
        print("Answer Bias Details:", answer_bias_details)
    print("\n" + "="*40 + "\n")

