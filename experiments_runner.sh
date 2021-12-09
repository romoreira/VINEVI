#!/bin/bash
echo "Running experiments this can take a while."
python3 hardware_monitoring.py
yourfilenames=`ls images_test`
for eachfile in $yourfilenames
do
   python3 load_example.py $eachfile
done
