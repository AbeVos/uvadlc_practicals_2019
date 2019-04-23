#!/bin/bash
for LENGTH in {4..35}
do
	echo "Input length: $LENGTH"
	./train.py --input_length $LENGTH --batch_size 512 --train_steps 2500 --model_type LSTM --learning_rate 0.01
done
