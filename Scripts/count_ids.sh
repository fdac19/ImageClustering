cat Data/ids.txt | while read id
do
	num=$(ls MORPH/Images/ | grep "$id" | wc -l)
	if [ $num -gt 29 ]; then
		echo "$id $num"
	fi
done
