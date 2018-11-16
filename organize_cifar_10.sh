filename="$1"
while read -r line; do
    name="$line"
    mkdir $name
    readarray -t arr < <(find -name "*$name*")
    for i in "${arr[@]}"
    do
		cp $i $name 
		# do whatever on $i
	done
done < "$filename"

# readarray -t arr < <(find -name "*sky*")
# #echo $arr[0],$arr[1]
# mkdir airplane
# mkdir 
# for i in "${arr[@]}"
# do
#    cp $i sky 
#    # do whatever on $i
# done
