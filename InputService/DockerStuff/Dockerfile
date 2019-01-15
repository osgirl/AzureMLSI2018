FROM tensorflow/serving:latest
RUN mkdir /MLServices
ADD ./requirements.txt ./MLServices/requirements.txt
ADD ./install.sh ./MLServices/install.sh



COPY . /MLServices
WORKDIR /MLServices
RUN ./install.sh
RUN CASS_DRIVER_NO_CYTHON=1 pip3 install -r ./requirements.txt
COPY ./models.py /usr/local/lib/python3.5/dist-packages/keras_vggface/
RUN ls .
ENTRYPOINT ["/bin/sh"]
CMD ["python mlservices.py"]
