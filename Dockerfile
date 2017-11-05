FROM bluelens/bl-cropper-base:latest

#ENV OD_MODEL ./model/frozen_inference_graph.pb
#ENV OD_LABELS ./model/label_map.pbtxt

#RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app

COPY . /usr/src/app

#RUN curl https://s3.ap-northeast-2.amazonaws.com/bluelens-style-model/object_detection/frozen_inference_graph.pb -o /usr/src/app/model/frozen_inference_graph.pb
#RUN curl https://s3.ap-northeast-2.amazonaws.com/bluelens-style-model/object_detection/label_map.pbtxt -o /usr/src/app/model/label_map.pbtxt

#RUN pip install -r requirements.txt
#RUN pip install --no-cache-dir google

CMD ["python", "main.py"]
