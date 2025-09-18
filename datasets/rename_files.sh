#!/bin/sh
#renames files of one extension to other
#relative path must be given for both:
#1 directory - path to the directory holding these files (not recursive)
#2 original extension
#3 change to extension

if [ -z "$1" ] || [ -z "$2" ] || [ "$1" = "" ] || [ "$2" = "" ] || [ -z "$3" ] || [ "$3" = "" ]
then
	echo "you all of 1=direcotry, 2=from extension, 3=to extension ... " 
	exit 1 
fi

cd $1

for f in *.$2; do
    mv -- "$f" "${f%.$2}.$3"
done
cd -
