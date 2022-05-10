#!/bin/bash

./chief.sh &
./tuner.sh tuner0 &
./tuner.sh tuner1 &
./tuner.sh tuner2 &
./tuner.sh tuner3 &