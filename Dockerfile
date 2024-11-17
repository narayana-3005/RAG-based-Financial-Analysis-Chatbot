FROM python:3.9-buster

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt requirements.txt 
COPY server.sh server.sh

# Set environment variables
ENV MLFLOW_SERVICE_PRIVATE_KEY='./secrets/credentials'

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port for MLflow
EXPOSE 5001

# Ensure server.sh is executable
RUN chmod +x server.sh

# Set the entry point to run the server script
ENTRYPOINT ["./server.sh"]
