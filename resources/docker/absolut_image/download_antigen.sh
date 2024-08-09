#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <antigen>"
    exit 1
fi

ANTIGEN=$1
INSTALL_DIR=/usr/local/Absolut

cd $INSTALL_DIR
url=$(./AbsolutNoLib info_filenames $ANTIGEN | grep -Eo 'https?://[^ ]+')
if [ -n "$url" ]; then
    echo "Downloading from URL: $url"
    filename=$(basename $url)
    wget $url -O $INSTALL_DIR/$filename
    unzip -o $INSTALL_DIR/$filename -d $INSTALL_DIR
    rm $INSTALL_DIR/$filename
else
    echo "No URL found for antigen: $ANTIGEN"
fi