FROM tensorflow/tensorflow:2.9.2-gpu

RUN mkdir -p src
ADD src /src

ENV PYTHONPATH "${PYTHONPATH}:/src"

RUN pip3 install scipy==1.8.1
RUN pip3 install pandas==1.4.2
RUN pip3 install pillow==9.1.1
RUN pip3 install aiohttp==3.8.1
RUN pip3 install aiofiles==0.8.0
RUN pip3 install psutil==5.9.1
RUN pip3 install scikeras==0.9.0
RUN pip3 install matplotlib==3.5.2
RUN pip3 install pydot==1.4.2
RUN pip3 install sklearn
RUN apt-get update -y
RUN apt-get install -y graphviz --fix-missing

ENTRYPOINT ["python3", "-u"] 