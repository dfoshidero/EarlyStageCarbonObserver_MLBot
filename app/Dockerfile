# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data using the script
RUN python download_nltk_data.py

# Download Spacy model
RUN python -m spacy download en_core_web_trf

# Ensure the sentence-transformers model is downloaded
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-mpnet-base-v2')"

# Expose port 80 to the outside world
EXPOSE 80

# Define environment variable
ENV NAME=Eco-Insight

# Run app.py when the container launches
CMD ["python", "app.py"]
