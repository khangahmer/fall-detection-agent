FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Headless agent (no UI):
CMD ["python", "fall_agent.py"]

# For the browser demo instead, override at run time:
#   docker run -p 8501:8501 <image> streamlit run app_streamlit.py --server.address=0.0.0.0
