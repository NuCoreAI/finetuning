#!/bin/sh
#removes sample-batches
#relative path must be given for both:
#1 directory - path to the directory holding these files (not recursive)
#2 batch NOT to be removed 
#3 the extension of the files (properties, commands, routines) 

if [ -z "$1" ] || [ -z "$2" ] || [ "$1" = "" ] || [ "$2" = "" ] || [ -z "$3" ] || [ "$3" = "" ]
then
	echo "you all of 1=direcotry, 2=batch NOT to be removed, 3=the extension of the files ... " 
	exit 1 
fi

for f in `ls ../customer_data/nodes/*.xml`; do
	uuid=$(basename $f .xml | awk -F '-' '{print $2;}')
	echo "processing ${uuid}"
	cd $1

	ls *${uuid}*${3}* | grep -v $2 | while read sample; do
			echo "removing $sample"
			rm -- "$sample"
	done
	cd -
done

