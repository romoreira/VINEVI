#!/bin/bash
echo "Running experiments this can take a while."

python3 hardware_monitoring.py "bittorrent" &
class="/bittorrent"
yourfilenames=`ls images_test$class`
echo "For each of Bittorrent image measuring prediction time"
for eachfile in $yourfilenames 
do
   python3 load_example.py $eachfile "bittorrent"
done
kill $(pgrep -f 'python3 hardware_monitoring.py')
kill $(pgrep -f 'python3 load_example.py')
echo "End of experiments Bittorrent"

python3 hardware_monitoring.py "browsing" &
class="/browsing"
yourfilenames=`ls images_test$class`
echo "For each of Browsing image measuring prediction time"
for eachfile in $yourfilenames
do
   python3 load_example.py $eachfile "browsing"
done
kill $(pgrep -f 'python3 hardware_monitoring.py')
kill $(pgrep -f 'python3 load_example.py')
echo "End of experiments Browsing"

python3 hardware_monitoring.py "dns" &
class="/dns"
yourfilenames=`ls images_test$class`
echo "For each of DNS image measuring prediction time"
for eachfile in $yourfilenames
do
   python3 load_example.py $eachfile "dns"
done
kill $(pgrep -f 'python3 hardware_monitoring.py')
kill $(pgrep -f 'python3 load_example.py')
echo "End of experiments DNS"

python3 hardware_monitoring.py "iot" &
class="/iot"
yourfilenames=`ls images_test$class`
echo "For each of IoT image measuring prediction time"
for eachfile in $yourfilenames
do
   python3 load_example.py $eachfile "iot"
done
kill $(pgrep -f 'python3 hardware_monitoring.py')
kill $(pgrep -f 'python3 load_example.py')
echo "End of experiments IoT"

python3 hardware_monitoring.py "rdp" &
class="/rdp"
yourfilenames=`ls images_test$class`
echo "For each of RDP image measuring prediction time"
for eachfile in $yourfilenames
do
   python3 load_example.py $eachfile "rdp"
done
kill $(pgrep -f 'python3 hardware_monitoring.py')
kill $(pgrep -f 'python3 load_example.py')
echo "End of experiments RDP"

python3 hardware_monitoring.py "ssh" &
class="/ssh"
yourfilenames=`ls images_test$class`
echo "For each of SSH image measuring prediction time"
for eachfile in $yourfilenames
do
   python3 load_example.py $eachfile "ssh"
done
kill $(pgrep -f 'python3 hardware_monitoring.py')
kill $(pgrep -f 'python3 load_example.py')
echo "End of experiments SSH"


python3 hardware_monitoring.py "voip" &
class="/voip"
yourfilenames=`ls images_test$class`
echo "For each of VOIP image measuring prediction time"
for eachfile in $yourfilenames
do
   python3 load_example.py $eachfile "voip"
done
kill $(pgrep -f 'python3 hardware_monitoring.py')
kill $(pgrep -f 'python3 load_example.py')
echo "End of experiments VOIP"
