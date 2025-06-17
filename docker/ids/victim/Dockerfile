FROM ubuntu:latest

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y \
    apache2 \
    openssh-server \
    vsftpd \
    samba \
    postfix \
    bind9 \
    ntp \
    tcpdump \
    iproute2 \
    curl \
    dnsutils \
    iputils-ping \
    netcat-openbsd && \
    apt-get clean

# SSH
RUN echo 'root:password' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# FTP
RUN sed -i 's/anonymous_enable=NO/anonymous_enable=YES/' /etc/vsftpd.conf

# Samba
RUN mkdir -p /smbshare && \
    chmod 0777 /smbshare && \
    echo -e "[public]\n   path = /smbshare\n   writable = yes\n   guest ok = yes\n   guest only = yes\n   create mode = 0777\n   directory mode = 0777\n" >> /etc/samba/smb.conf && \
    smbpasswd -a -n nobody

# NTP: use default

# BIND9（DNS): use default

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 80 21 22 25 53 123 139 445

CMD ["/start.sh"]