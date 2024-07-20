#!/bin/bash -e

# power on and connect the husky
docker-compose up -d

sudo ifconfig enp108s0 192.168.3.100
sudo route add 192.168.1.201 enp108s0

