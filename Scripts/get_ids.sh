ls MORPH/Images/ | while read id
do
	echo "$id" | sed 's/_.*//g'
done
