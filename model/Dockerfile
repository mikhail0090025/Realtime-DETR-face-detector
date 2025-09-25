FROM my-base-image:latest

WORKDIR /app

COPY ./model /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
