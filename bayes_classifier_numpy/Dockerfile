FROM ubuntu:18.04
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q python-all python-pip
RUN apt-get install -y python3.8 \
                       python3.8-distutils \
		       python3-pip \
                       git \
                       curl \
		       libblas-dev \
		       liblapack-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
WORKDIR /opt/bayes_classifier_numpy
RUN pip3 install --upgrade pip
RUN git clone --depth 1 https://github.com/aimacode/aima-python.git .
RUN git fetch --unshallow 
COPY ../files/requirements /tmp/requirements
RUN pip3 install -r /tmp/requirements
RUN python3 -m nltk.downloader stopwords
ADD ./bayes_classifier_numpy/bayes_classifier_numpy/ /opt/bayes_classifier_numpy/
COPY ./../files/ /opt/bayes_classifier_numpy/
EXPOSE 5001
CMD ["python3", "bayes_classifier_numpy.py"]

