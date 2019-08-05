#!/bin/bash

DIR="data/archives"
cd $DIR
for file in *.tgz;
do
    tar -xzvf $file --wildcards '*.wav' --strip-components=2 --one-top-level=`echo $file | sed -E 's/(.+).tgz/\1/g'`
done 


# string="twet.tgz"
# s=`echo $string | sed -r 's/Seconds_Behind_Master: ([0-9]+)/\1/g'`
# s2=`echo $string | sed -r 's/(.+).tgz/\1/g'`
# echo $s2