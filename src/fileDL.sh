#!/bin/bash

#example download for San Francisco 
#curl -O https://www.minneapolisfed.org/news-and-events/beige-book-archive/2018-07-sf

#base URl for files
base="https://www.minneapolisfed.org/news-and-events/beige-book-archive/"


#single touch/request version (last 5 years)
#with -O option this will write to current directory (overwriting if necessary)
curl -Os "$base[2014-2018]-{01,04,07,10}-{at,bo,ch,cl,da,kc,mi,ny,ph,ri,sf,sl,su}"
#remove 's' (silent) to see progress meter


#multi request version
#declare array of district names: 12 + 1 national summary
declare -a distnames=(
#					"at" #Atlanta
#  					"bo" #Boston
#  					"ch" #Chicago
#  					"cl" #Cleveland
#  					"da" #Dallas
#  					"kc" #Kansas City
#  					"mi" #Minneapolis
#  					"ny" #NY
#  					"ph" #Philadelphia
#  					"ri" #Richmond
  					"sf" #SF
#  					"sl" #St. Louis
#  					"su" #National Summary
 					)

#array of years required 
declare -a fYears=("2018" "2017" "2016" "2015" "2014")

#array of months (8 every year, but not the same months). Always one at start of Qtr?
declare -a fMths=("01" "04" "07" "10")

#12 (districts) * 8 (reports/year) * 49 (1970-2018) = nearly 5000 reports

#download the data files sequentially (multiple connections, handshakes, yikes..).
for i in "${distnames[@]}"
do
	for y in "${fYears[@]}"
	do
		for m in "${fMths[@]}"
		do
			#in Bash safer to use quotes "$i" when var may have spaces or shell expandable chars"		
			echo "downloading file " "$y"-"$m"-"$i"
			#download and move to file in reports directory
			curl "$base""$y"-"$m"-"$i" > ../reports/"$y"-"$m"-"$i"".html"
		done
	done
done


#echo "ensure proper directory before downloading"

#move files to S3
#echo "Moving files to S3"

# for f in "${fnameArr[@]}"
# do
# 	aws s3 cp  ~/data/"$f" s3://lendclub/data/"$f"
# done

