FROM bluelens/tensorflow:1.3.0-py3

ENV OD_MODEL ./model/frozen_inference_graph.pb
ENV OD_LABELS ./model/label_map.pbtxt

RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install -r requirements.txt
RUN pip install --no-cache-dir google

CMD ["python", "main.py"]
