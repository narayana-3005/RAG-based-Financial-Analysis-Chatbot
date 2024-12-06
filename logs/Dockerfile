# Use an official Python runtime as a parent image
FROM python:3.8-slim

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
