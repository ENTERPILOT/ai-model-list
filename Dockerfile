FROM python:3.12-slim
WORKDIR /app
COPY requirements-editor.txt .
RUN pip install --no-cache-dir -r requirements-editor.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "editor.app:app", "--host", "0.0.0.0", "--port", "8000"]
