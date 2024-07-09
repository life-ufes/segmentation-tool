FROM nvcr.io/nvidia/pytorch:24.05-py3


# Update image
RUN apt update
WORKDIR /app
ENV TZ=America \
    DEBIAN_FRONTEND=noninteractive

# RUN apt install -y  ffmpeg libsm6 libxext6 -y



# Install pip packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache -r requirements.txt
RUN pip install opencv-python 
RUN pip install opencv-fixer==0.2.5
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"


WORKDIR /app