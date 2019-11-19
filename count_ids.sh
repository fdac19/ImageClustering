cat ids.txt | while read id
do
	num=$(grep "$id" ids.txt | wc -l)
	if [ $num -gt 29 ]; then
		echo "$id $num"
	fi
done
