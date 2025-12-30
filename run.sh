#!/bin/bash
# Konversi path saat ini ke format Windows untuk Python Windows
WIN_PATH=$(cygpath -w $(pwd))
export PYTHONPATH=$PYTHONPATH:$WIN_PATH
echo "PYTHONPATH set to $WIN_PATH"
python tools/run_net.py "$@"
