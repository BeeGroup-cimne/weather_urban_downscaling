FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /app
RUN apt-get update && apt-get install -y git graphviz && rm -rf /var/lib/apt/lists/*
COPY requirements_tf.txt .
RUN pip install --no-cache-dir -r requirements_tf.txt
COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
CMD ["python", "scripts/run_ablation.py"]
