cat best_ids.txt | while read id
do 
	ls MORPH/Images/ | grep "$id"
done
