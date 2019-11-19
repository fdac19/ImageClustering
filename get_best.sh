cat best_ids.txt | while read id
do 
	grep "$id" image_paths.txt
done
