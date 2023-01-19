FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
WORKDIR app/
CMD python -m uvicorn --reload --port 8000 main:app
