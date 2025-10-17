#!/bin/sh
#Replaces text in files of certain pattern
#1. relative path to the directory
#2. the pattern of files to look at 
#3. original text to be replaced. If in quotes make sure to escape properly. i.e. '"old text"'
#3. original text to be replaced. If in quotes make sure to escape properly. i.e. '"new text"'

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" = "" ] || [ "$4" = "" ]
then
	echo "you all of 1=direcotry, 2=file pattern, 3=original text, 4=new text ... " 
	exit 1 
fi

cd $1

for f in $2; do
	echo "$f ..." 
	sed -i "s/$3/$4/g" "$f"
done
cd -
