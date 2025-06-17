#!/bin/bash

TARGET_IP=${VICTIM_IP:-victim}
INTERVAL=${INTERVAL:-1}

DNS_SERVER=${DNS_SERVER:-8.8.8.8}
NTP_SERVER=${NTP_SERVER:-pool.ntp.org}
MAIL_SERVER=${MAIL_SERVER:-${TARGET_IP}} 

echo "Start to generate various normal traffics: target ${TARGET_IP}, interval ${INTERVAL} seconds"

while true; do
  # HTTP/HTTPS
  echo "Testing HTTP/HTTPS connectivity to ${TARGET_IP}"
  curl -s -o /dev/null -w "%{http_code}\n" http://${TARGET_IP}/
  curl -s -o /dev/null -w "%{http_code}\n" https://${TARGET_IP}/

  # DNS
  echo "Testing DNS resolution for example.com using ${DNS_SERVER}"
  dig example.com @$DNS_SERVER > /dev/null

  # NTP
  echo "Testing NTP synchronization with ${NTP_SERVER}"
  ntpdate -q $NTP_SERVER > /dev/null 2>&1

  # SSH
  echo "Testing SSH connectivity to ${TARGET_IP}"
  timeout 2 bash -c "cat < /dev/null > /dev/tcp/${TARGET_IP}/22"

  # SMTP
  echo "Testing SMTP connectivity to ${MAIL_SERVER}"
  timeout 2 bash -c "cat < /dev/null > /dev/tcp/${MAIL_SERVER}/25"

  # FTP
  echo "Testing FTP connectivity to ${TARGET_IP}"
  timeout 2 bash -c "cat < /dev/null > /dev/tcp/${TARGET_IP}/21"

  # SMB
  echo "Testing SMB connectivity to ${TARGET_IP}"
  timeout 2 bash -c "cat < /dev/null > /dev/tcp/${TARGET_IP}/445"

  # ICMP
  echo "Testing ICMP connectivity to ${TARGET_IP}"
  ping -c 2 ${TARGET_IP}

  SLEEP_TIME=$(shuf -i ${INTERVAL}-$((${INTERVAL}+3)) -n 1)
  sleep ${SLEEP_TIME}
done