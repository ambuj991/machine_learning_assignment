FROM python:3.9-slim
# Install build dependencies

RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    build-essential


WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir h5py==3.11.0

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command to run your application
CMD ["python", "model.py"]
