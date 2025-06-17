FROM alpine:latest

RUN apk update && \
    apk add --no-cache \
    bash \
    curl \
    iputils \
    wget \
    tcpdump \
    nmap \
    bind-tools \
    openntpd \
    coreutils

RUN mkdir -p /scripts

CMD ["/bin/bash"]
