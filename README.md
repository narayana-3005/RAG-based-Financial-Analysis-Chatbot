# RAG-based-Financial-Analysis-Chatbot
In the rapidly evolving financial landscape, access to accurate and timely information is paramount for informed decision-making. To address this need, we propose the development of a Retrieval-Augmented Generation (RAG) based chatbot tailored for Quality Assurance (QA) in the financial sector. This chatbot will leverage comprehensive datasets comprising real-time stock prices and financial news to provide users with precise answers, insightful analysis, and up-to-date information. By integrating advanced natural language processing (NLP) techniques with robust data sources, our solution aims to enhance user experience, streamline information retrieval, and support financial professionals in making data-driven decisions.

RAG Based Financial Chatbot 

# Introduction: 

 

In the rapidly evolving financial landscape, access to accurate and timely information is paramount for informed decision-making. To address this need, we propose the development of a Retrieval-Augmented Generation (RAG) based chatbot tailored for Quality Assurance (QA) in the financial sector. This chatbot will leverage comprehensive datasets comprising real-time stock prices and financial news to provide users with precise answers, insightful analysis, and up- to-date information. By integrating advanced natural language processing (NLP) techniques with robust data sources, our solution aims to enhance user experience, streamline information retrieval, and support financial professionals in making data-driven decisions. 

 
# Dataset Information: 

 

The foundation of our RAG-based chatbot relies on two primary datasets: stock prices and financial news. 

• Stock Prices: This dataset includes historical and real-time stock market data such as open, close, high, low prices, trading volumes, and other relevant financial indicators. It serves as the quantitative backbone for analyzing market trends and performing financial forecasting. 

• Financial News: This data set comprises articles, press releases, and reports from  

reputable financial news outlets. It provides qualitative insights into market sentiments,  

company performances, economic indicators, and other factors influencing stock prices. 

These datasets are integral to the project as they enable the chatbot to deliver comprehensive and contextually relevant responses, merging numerical data with narrative information to support robust QA functionalities in the financial domain. 


# Data Source: 

 

To ensure the reliability and comprehensiveness of our datasets, we have identified the following  

sources: 

Stock Prices: 

• Yahoo Finance API: Yahoo Finance API 

• Alpha Vantage: Alpha Vantage API 

• Quandl: Quandl Financial Data 

Financial News: 

• Bloomberg: Bloomberg News 

• Reuters: Reuters Financial News 

• NewsAPI: NewsAPI (for aggregated financial news) 

These sources provide high-quality, up-to-date information essential for the chatbot's  

functionality. APIs offer seamless integration for real-time data fetching, while web scraping  

techniques will be employed where necessary to gather comprehensive news articles. 
Flow Chart for the approach <img width="962" alt="Screenshot 2024-10-01 at 9 49 50 PM" src="https://github.com/user-attachments/assets/5db6837d-094b-471e-b194-c07248fd35f8">

# Tools: 

Cloud Run 

Cloud Run hosts containerized applications, making it ideal for deploying the chatbot's API and backend services. 

Purpose: Serves as the primary deployment platform for the chatbot’s REST API and MLflow server. 

Features: 

Automatic scaling based on incoming requests. 

Fully managed, serverless environment. 

Workflow: 

The Dockerized chatbot application is deployed on Cloud Run. 

It processes user queries, retrieves relevant data, and generates responses using fine-tuned language models. 

 

# Cloud Functions 

Cloud Functions handle lightweight serverless computations, focusing on event-driven tasks like data ingestion and preprocessing. 

Purpose: Automates the ETL process of data ingestion from external APIs (e.g., Yahoo Finance) and preprocessing pipelines. 

Workflow: 

Triggered by events such as new data uploads to GCS. 

Executes operations like data cleaning, transformation, and storage in GCS for further use. 

Each bucket is filled with new data with these and triggers the next dependent bucket and cloud function. 

 

Cloud Build 

Cloud Build automates the CI/CD pipeline for the project. 

Purpose: Ensures a seamless build, test, and deployment workflow for Docker images and application updates. 

Workflow: 

Monitors code repositories for changes. 

Builds the Docker image and pushes it to Artifact Registry. 

Triggers deployment to Cloud Run or Cloud Functions. 

 

# Artifact Registry 

Artifact Registry stores and manages Docker images and other artifacts. 

Purpose: Provides a secure location for storing Docker images used by Cloud Run and Cloud Functions. 

Workflow: 

Stores containerized versions of the chatbot and preprocessing scripts. 

Ensures version control and easy access during deployment. 

Secret Manager 

Secret Manager securely handles sensitive information such as API keys, database credentials, and service account keys. 

Purpose: Prevents hardcoding of secrets in the application and ensures secure access. 

Workflow: 

Stores critical secrets like GCS service account keys and PostgreSQL credentials. 

Grants controlled access to these secrets via IAM policies to Cloud Run and Cloud Functions. 

 

MLflow 

MLflow is used for tracking machine learning experiments, logging models, and managing deployment workflows. 

Purpose: Tracks all experiment metadata, including parameters, metrics, and artifacts. 

Workflow: 

MLflow logs training runs, model versions, and evaluation metrics. 

Artifacts such as trained models and logs are stored in GCS. 

The MLflow tracking server is hosted on Cloud Run. 

 

Google Cloud Storage (GCS) 

GCS buckets are used for storing raw and processed data, as well as MLflow artifacts. 

Purpose: Centralized storage solution for unstructured raw stock data, preprocessed datasets, and trained model files. 

Workflow: 

Stores input datasets (e.g., stock prices, news articles). 

Hosts model artifacts and logs for retrieval by MLflow and the chatbot API. 

 

# Data Preprocessing: 

Cleaning  

• Stock Prices: Remove duplicates, handle missing values, and convert timestamps to a standard datetime format. 

 • Financial News: Normalize text data by converting to lowercase, removing special characters, and eliminating stop words using libraries like NLTK or spaCy. 

Feature Engineering  

• Create features like moving averages for stock data; extract sentiment and key entities from news articles. 

Text Processing  

• Normalize text (lowercase, remove special characters) and tokenize for NLP tasks. 

 Normalization  

• Normalize numerical features from the stock prices to ensure they are on a similar scale (e.g., using Min-Max scaling) 

 

# Data Pipeline Orchestration: 

 

The first step in building a robust data pipeline is setting up Apache Airflow. Airflow is an open-source platform designed for programmatically authoring, scheduling, and monitoring workflows. It empowers users to orchestrate complex data pipelines with reliability and ease. 

 

 

Step 1: Install Apache Airflow 

To install Apache Airflow, execute the following command in your environment: 

pip install apache-airflow 
 

Step 2: Launch the Airflow Scheduler and Web Server 

Once installed, initialize and start Airflow's scheduler and web server. This allows you to manage Directed Acyclic Graphs (DAGs) through a browser-based User Interface (UI). DAGs define tasks and their dependencies, enabling seamless workflow automation. 

Run the following commands: 

Initialize the Airflow Database: 

airflow db init 

Start the Scheduler: 

airflow scheduler 

Launch the Web Server: 

airflow webserver 
 

Step 3: Access the Airflow UI 

Navigate to the Airflow web UI by opening http://localhost:8080 in your browser. Here, you can define, monitor, and manage your workflows interactively. 

 

# Features: 

 

Tracking 

 

Logging 

 

DVC 

 

Experimentation 

 

Monitoring 

 

Load Balancing 

 

 

# CI-CD 

A new Docker image is created for both chatbot training and deployment whenever there is a push to the main branch. This process is automated using GitHub Actions, with the build-and-deploy.yaml file in the Actions folder managing the image build and push workflows. 

To enhance efficiency, future improvements could involve creating a Docker image only when changes are made to the chatbot training or deployment code, thereby avoiding unnecessary builds. 

The code in the airflow-dags-financial-chatbot bucket remains synchronized with the main branch, ensuring that Cloud Composer always executes the latest DAGs in Airflow. This synchronization is achieved by configuring a Cloud Build Trigger in Google Cloud Platform (GCP) and maintaining a corresponding cloudbuild.yaml file in the GitHub repository. 

This setup ensures that the financial chatbot's workflows and Airflow tasks remain up to date, enabling seamless automation and synchronization. Let me know if further modifications are needed! 

 

# Model 

 

Our model integrates advanced Large Language Model (LLM) capabilities with Retrieval-Augmented Generation (RAG) techniques to deliver real-time financial analyses and insights. Leveraging a combination of Dense Passage Retrieval (DPR) and Facebook AI Similarity Search (FAISS), the system ensures efficient document retrieval and precise QA generation. The architecture includes: 

DPR Context and Question Encoders to convert context documents and user queries into dense vector representations, enabling high-quality semantic matching. 

FAISS Index, constructed using encoded context embeddings, ensures fast and scalable retrieval of top-k relevant documents, achieving real-time performance on large datasets. 

DPR Reader, which processes retrieved documents and user queries to generate precise answers by identifying the start and end tokens within the context. 

To optimize performance, we employed Low-Rank Adaptation (LoRA) and model quantization, reducing memory usage and enhancing inference speed without compromising the quality of generated insights. The model undergoes rigorous testing with unit tests to validate document retrieval accuracy, answer generation, and FAISS index integrity. 

The pipeline is integrated with Google Cloud Composer, where successful data preprocessing triggers model training. The training workflow is containerized in a Docker image and deployed using Google Cloud Run, ensuring scalability and seamless updates. The Docker image is rebuilt and updated automatically with each push to the main branch of the financial-chatbot GitHub repository. Upon completing training, the updated model parameters are pushed to Hugging Face, ensuring the latest version is readily available for deployment. 

This system’s robust and scalable architecture makes it ideal for real-time question answering, customer support, dynamic financial analysis, and knowledge retrieval, delivering state-of-the-art solutions with cutting-edge machine learning and natural language processing techniques. 

Model Components 

 

These are tasks that form the core of the model training pipeline, leveraging Retrieval-Augmented Generation (RAG) techniques to deliver high-performance financial analysis and insights. Below is a detailed overview of each task: 

load_data_from_gcs: 

Retrieves preprocessed JSON data from a Google Cloud Storage (GCS) bucket using the storage.Client() from the Google Cloud SDK. The data serves as input for training and evaluation processes. 

upload_to_gcs: 

Uploads generated files, such as training metrics, evaluation results, and performance plots, to a specified GCS bucket using the storage.Client() from the Google Cloud SDK. 

load_model_tokenizer: 

Initializes the base model and tokenizer required for training. The configuration includes model specifications such as sequence length, data type for tensors (auto-detected), and quantization settings for resource efficiency. 

load_peft_model: 

Implements Low-Rank Adaptation (LoRA) using PEFT (Parameter-Efficient Fine-Tuning) configurations. This step enhances the base model by applying LoRA-based optimizations, such as adjusting dropout rates, gradient checkpointing for handling long sequences, and fine-tuning targeted model layers. 

prepare_data: 

Prepares and tokenizes training data for natural language processing tasks. The process involves splitting the data into training and test sets, applying prompt engineering for RAG tasks (e.g., summarization or retrieval-based QA), and ensuring compatibility with model requirements. 

train_model: 

Executes model training using the Supervised Fine-Tuning Trainer (SFTTrainer) with predefined arguments and settings. During training, key metrics such as loss values are logged, and training loss plots are generated and uploaded to GCS for monitoring. 

evaluate_model: 

Assesses the model’s performance on test data using metrics like ROUGE-L score and semantic similarity to ensure the quality of generated responses. Evaluation outputs, including detailed graphs, are saved to GCS for further analysis and validation. 

containerization and deployment: 

The model is containerized using Docker and deployed via Google Cloud Run for seamless scalability and accessibility. The deployment pipeline integrates with Google Cloud Composer for task orchestration, ensuring end-to-end automation. Upon successful training, the updated model is deployed for production use, optimized for real-time query handling in financial analysis scenarios. 

 

 

# Model Evaluation: 

 

 

# Model Deployment: 

 

 

# Drift Detection: 

 

 

# High Level End-to-End Design Overview 

 

 Data Ingestion: 

Cloud Functions fetch stock data from APIs like Yahoo Finance or Polygon.io and store it in GCS. 

Preprocessing pipelines clean and organize the data. 

Experimentation: 

MLflow tracks experiments, storing metadata in PostgreSQL and artifacts in GCS. 

Trained models are validated and logged for deployment. 

Model Deployment: 

Models are containerized and stored in the Artifact Registry. 

Deployed to Cloud Run as APIs for real-time chatbot interactions. 

Secrets Management: 

Sensitive credentials are securely stored in Secret Manager and accessed dynamically by the application. 

User Interaction 

Building Financial Chatbot: 
After integrating Retrieval-Augmented Generation (RAG) techniques, the chatbot processes financial data for real-time query responses. It utilizes DPR and FAISS for fast, context-based document retrieval and precise answer generation. 

Data Handling Through API Calls: 
Once data is scraped, it is sent to an intermediary server via API calls. The server stores this data in Google Cloud Storage, making it available for future model training and evaluation. 

Model Integration: 
After the data is stored, the server forwards it to the model for processing. The model uses the stored data to generate context-specific, summarized responses tailored to financial queries 


