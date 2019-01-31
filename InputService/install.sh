#! /bin/sh

apt update && apt install -y python3 python3-pip libsm6 libxext6 libfontconfig1 libxrender1 libglib2.0-0
pip3 install --upgrade pip