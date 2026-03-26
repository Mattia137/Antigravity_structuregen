FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Hugging Face Spaces requires the app to run on port 7860
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "server:app"]
