FROM python:3
RUN mkdir /VisualizationService
ADD ./requirements.txt ./VisualizationService/requirements.txt
#CASS_DRIVER_NO_CYTHON speeds up compilation for testin
RUN CASS_DRIVER_NO_CYTHON=1 pip install -r ./VisualizationService/requirements.txt

COPY . /VisualizationService
WORKDIR /VisualizationService
RUN ls .
ENTRYPOINT ["python"]
CMD ["VisualizationService.py"]
