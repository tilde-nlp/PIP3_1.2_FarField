FROM python:3.9

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt

ENV SSH_PASSWD "root:Docker!"
RUN apt-get update \
        && apt-get install -y --no-install-recommends dialog \
        && apt-get update \
	&& apt-get install -y --no-install-recommends openssh-server \
	&& echo "$SSH_PASSWD" | chpasswd 

RUN apt-get install -y libsndfile1

RUN ls

COPY ./sshd_config /etc/ssh/

EXPOSE 8080 2222

ENTRYPOINT ["python", "main.py"]