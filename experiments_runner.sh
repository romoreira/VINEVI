#!/bin/bash
echo "Running experiments this can take a while."
python3 hardware_monitoring.py $1 &
class="/$1"
yourfilenames=`ls images_test$class`
echo "For each image measuring prediction time"
for eachfile in $yourfilenames 
do
   echo "$eachfile"
   python3 load_example.py $eachfile $1
done
kill $(pgrep -f 'python3 hardware_monitoring.py')
kill $(pgrep -f 'python3 load_example.py')
echo "End of experiments"
