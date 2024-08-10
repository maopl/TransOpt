#!/bin/bash
pip install transopt_external/csstuning

bash transopt_external/csstuning/cssbench/compiler/docker/build_docker.sh
bash transopt_external/csstuning/cssbench/dbms/docker/build_docker.sh

csstuing_dbms_init -h
