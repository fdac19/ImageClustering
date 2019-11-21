cat ../Data/best_ids30.txt | while read id
do 
	ls ../MORPH/Images/ | grep "$id"
done
