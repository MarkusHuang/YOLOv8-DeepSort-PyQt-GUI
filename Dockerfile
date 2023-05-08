FROM continuumio/anaconda3

LABEL image.title="YOLOv8 GUI" \
      image.description="YOLOv8 GUI container" \
      image.authors="JingH"

# Create working directory
RUN mkdir -p /usr/src/YOLOv8-DeepSort-PyQt-GUI
WORKDIR /usr/src/YOLOv8-DeepSort-PyQt-GUI

# 
RUN git clone https://github.com/MarkusHuang/YOLOv8-DeepSort-PyQt-GUI.git /usr/src/YOLOv8-DeepSort-PyQt-GUI

# Create the conda environment
RUN conda env create -f environment.yml

# Activate the env so that the next packages are installed in it
SHELL ["conda", "run", "-n", "yolov8_gui", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
      libgl1 libxkbcommon-x11-0\
      && rm -rf /var/lib/apt/lists/* \
      # Demonstrate the environment is activated:
      && echo "Make sure numpy is installed:"\
      && python -c "import numpy"

# Download weights
RUN python download_weights.py
