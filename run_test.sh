#!/bin/bash

if [ ! -f test_data.bin ]; then
  cp ./data/test_data.bin ./test_data.bin
fi

if [ ! -d networks ]; then
  echo 'run download_pack.sh before running run_test.sh'
else
  ./build/test_networks
fi
