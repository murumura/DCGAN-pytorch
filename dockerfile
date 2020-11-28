FROM pytorch/pytorch
RUN apt-get update && apt-get install -y --no-install-recommends git
RUN apt-get install python3-pip -y
RUN apt-get install libsndfile1 -y
RUN pip install --upgrade pip
RUN pip3 --no-cache-dir install \
    Pillow \
    h5py \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    future \
    portpicker\
    librosa\
    numba==0.48.0\
    tqdm\
    scikit-image\
    notebook
    # Add Tini. Tini operates as a process subreaper for jupyter. This prevents
WORKDIR /torch
# kernel crashes.
EXPOSE  8888
EXPOSE  6666
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/torch --ip 0.0.0.0  --allow-root"]

