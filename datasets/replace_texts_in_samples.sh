#!/bin/sh
#Replaces texts in files of certain pattern
#1. relative path to the directory
#2. the pattern of files to look at 

if [ -z "$1" ] || [ -z "$2" ] 
then
	echo "you need all of 1=direcotry, 2=file pattern ... " 
	exit 1 
fi

./replace_text_in_samples.sh "$1" "$2" '"name of ambiguous event 1"}' '"name of ambiguous event 1\\"}'
./replace_text_in_samples.sh "$1" "$2" '"name of ambiguous event 2"}' '"name of ambiguous event 2\\"}'
./replace_text_in_samples.sh "$1" "$2" '"name of possible device 1"}' '"name of possible device 1\\"}'
./replace_text_in_samples.sh "$1" "$2" '"name of possible device 2"}' '"name of possible device 2\\"}'
./replace_text_in_samples.sh "$1" "$2" 'contextal' 'contextual'
./replace_text_in_samples.sh "$1" "$2" 'Aways' 'Always'
./replace_text_in_samples.sh "$1" "$2" '=! (is not)' '!= (is not), is, is not' 
./replace_text_in_samples.sh "$1" "$2" 'IS' '==' 
./replace_text_in_samples.sh "$1" "$2" '=!' '!=' 

