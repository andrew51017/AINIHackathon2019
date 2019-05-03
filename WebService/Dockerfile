FROM gw000/keras:2.1.4-py3-tf-cpu

ADD main.py /
ADD model.json /
ADD model.h5 /
ADD tokenizer.pickle /

RUN pip install flask

CMD [ "python", "./main.py" ]