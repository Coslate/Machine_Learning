#! /bin/csh -f

./hw4.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz --infile_train_image ./sample/train-images-idx3-ubyte.gz -pse_p 0.00001666666 -pse_lamd 0.1  --use_pseudo_w 1 --use_pseudo_lambda 0 --use_pseudo_p 0 -tcn 35 -isd 0
