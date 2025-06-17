#!/bin/bash

service apache2 start
service ssh start
service vsftpd start
service smbd start
service nmbd start
service postfix start
service bind9 start
service ntp start

tcpdump -i eth0 -w /pcap/capture.pcap &
TCPDUMP_PID=$!

sleep 14400

kill $TCPDUMP_PID

tail -f /dev/null
