FROM ubuntu:latest

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y nmap iputils-ping iproute2 hping3 apache2-utils nikto && \
    apt-get clean

CMD ["sh", "-c", "tail -f /dev/null"]
