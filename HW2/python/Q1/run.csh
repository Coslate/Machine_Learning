#! /bin/csh -f


./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./ungzip_sample/t10k-images.idx3-ubyte --toggle 0 -isd 1 > ./ans.log

# ./hw2.py --infile_train_label ./ungzip_sample/train-labels.idx1-ubyte  --infile_train_image ./ungzip_sample/train-images.idx3-ubyte --infile_test_label ./ungzip_sample/t10k-labels.idx1-ubyte --infile_test_image ./ungzip_sample/t10k-images.idx3-ubyte --toggle 0 -isd 1 > ./ans.log

