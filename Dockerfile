# Use Python with Windows compatibility
FROM python:3.9-windowsservercore-ltsc2019

# Set working directory
WORKDIR C:/app

# Install system dependencies
RUN pip install --upgrade pip

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]