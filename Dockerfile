FROM python:3.9-buster

# Set the working directory
WORKDIR /app

# Install gcloud CLI
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk

# Install pip dependencies including mlflow
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy necessary files
COPY server.sh server.sh

# Expose the port for MLflow
EXPOSE 5001

# Ensure server.sh is executable
RUN chmod +x server.sh

# Set entrypoint
ENTRYPOINT ["./server.sh"]
