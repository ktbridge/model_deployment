# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the requirements first to leverage Docker caching
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code and assets
COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/
COPY tests/ /app/tests/

# 6. Expose the port FastAPI will run on
EXPOSE 8000

# 7. Command to run the Uvicorn server inside the container
CMD ["uvicorn", "src.devops_ml.app:app", "--host", "0.0.0.0", "--port", "8000"]