FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY clients ./clients
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "fl_async.main:app", "--host", "0.0.0.0", "--port", "8000"]
