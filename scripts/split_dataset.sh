#!/bin/bash

total=$(wc -l < ~/datasets/drsk/image_names.txt)
train_count=$(($total * 9 / 10))
shuf ~/datasets/drsk/image_names.txt > ~/datasets/drsk/image_names_shuffled.txt
head -n $train_count ~/datasets/drsk/image_names_shuffled.txt > ~/datasets/drsk/image_names_train.txt
tail -n +$(($train_count + 1)) ~/datasets/drsk/image_names_shuffled.txt > ~/datasets/drsk/image_names_eval.txt
rm ~/datasets/drsk/image_names_shuffled.txt