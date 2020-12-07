#!/bin/sh -e

# Collect categorical labels of IEMOCAP
# Labels can be obtained from files in
# Session?/dialog/EmoEvaluation/Ses0??_impro0?.txt
# Session?/dialog/EmoEvaluation/Ses0??_script0?_?.txt

# Define constants
corpusdir=/corpus/iemocap
label_file="../labels/emo_labels.txt"

echo -n "" > $label_file
for i in `seq 5`
do
    txt_files=`find $corpusdir/Session${i}/dialog/EmoEvaluation/Categorical/* -maxdepth 0 -type f -name "*.txt"`
    for file in $txt_files
    do
	cat $file | grep Ses >> $label_file
    done
done
