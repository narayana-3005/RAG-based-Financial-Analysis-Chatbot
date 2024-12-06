# Use an official Python runtime as a parent image
FROM python:3.8-slim



# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set environment variables for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /logs

# Copy the current directory contents into the container
COPY . /logs

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8085 available to the world outside this container
EXPOSE 8085

# Run app.py when the container launches
CMD ["python", "logs.py"]
