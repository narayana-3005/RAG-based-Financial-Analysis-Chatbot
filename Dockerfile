# Use Python 3.9 slim image as the base
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and enable buffer flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file to leverage Docker's caching mechanism
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose the application port (if your application runs on a specific port, e.g., 8080)
EXPOSE 8080

# Set the default command to run your application
CMD ["python", "stock_prices_transformed_cloud_function.py"]  # Replace "app.py" with the entry point script of your project
