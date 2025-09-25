FROM my-base-image:latest

WORKDIR /app

COPY ./frontend /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]