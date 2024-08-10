#!/bin/bash
git clone https://github.com/neeetman/csstuning.git && cd csstuning
pip install .

bash cssbench/compiler/docker/build_docker.sh
bash cssbench/dbms/docker/build_docker.sh

csstuing_dbms_init -h