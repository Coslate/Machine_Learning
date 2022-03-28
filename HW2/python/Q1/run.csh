#! /bin/csh -f

#############################Discrete Mode##############################
#Error rate: 0.1486
./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 0 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 0.0001 --image_method_disc 1 --use_color 1

./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 0 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 0.0001 --image_method_disc 1 --use_color 0 > ./ans.disc.log

#Error rate: 0.1486
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 0 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 0.0001 --method 0 --d_gauss 1 --image_method_disc 1

#Error rate: 0.1493
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 0 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 0.00001--method 0 --d_gauss 1 > ./ans.log

#Error rate: 0.1487
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 0 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 0.000045 --method 0 --d_gauss 1 > ./ans.log

#Error rate: 0.1583
./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 0 -isd 0 --pseudo_cnt_method 0 --PSEUDO_CNST 0.0001 --image_method_disc 1 --use_color 1


#############################Continuous Mode##############################
#---------------------------------------method 0--------------------------------------#
#Error rate: 0.2021
./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300  --use_color 1

./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --use_color 0 > ./ans.cont.log

#Error rate: 0.2026
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2400


#Error rate: 0.2033
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2500


#Error rate: 0.2023
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2200

#Error rate: 0.2022
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2100

#Error rate: 0.2029
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2000

#Error rate: 0.3517, by skipping pixels with var==0
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 0 --PSEUDO_CNST 2300

#---------------------------------------method 1--------------------------------------#
#Error rate: 0.2039
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 0

#Error rate: 0.201
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 1

#Error rate: 0.1977
./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 2

#Error rate: 0.192
./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 4 > ./ans.cont.log

#Error rate: 0.1874
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 8

#Error rate: 0.1806
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 16

#Error rate: 0.1763
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 32

