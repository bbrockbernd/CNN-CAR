#!/bin/bash

export KERASTUNER_TUNER_ID="$1"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"

python Model2D.py > $1.txt