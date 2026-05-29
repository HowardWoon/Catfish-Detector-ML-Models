FROM python:3.11-slim

# Set up working directory
WORKDIR /code

# Install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the files
COPY . .

# Hugging Face Spaces requires exposing port 7860
EXPOSE 7860

# We use gunicorn to serve the Flask app, giving it 120 seconds timeout for heavy ML model loading
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-t", "120", "webapp.app:create_app()"]
