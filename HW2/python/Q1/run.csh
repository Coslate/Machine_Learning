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

#############################Continuous Mode##############################
#---------------------------------------method 0--------------------------------------#
#Error rate: 0.2021
./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300  --use_color 1

./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --use_color 0 > ./ans.cont.log

#Error rate: 0.2021
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 0 --d_gauss 0 --use_color 1

#Error rate: 0.2026
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2400 --method 0 --d_gauss 0


#Error rate: 0.3517, by skipping pixels with var==0
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 0 --PSEUDO_CNST 2300 --method 0 --d_gauss 0


#Error rate: 0.4391, by using min var of other pixel within same digit
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 0 --PSEUDO_CNST 2300 --method 0 --d_gauss 0


#---------------------------------------method 2--------------------------------------#
#Error rate: 
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 2 --d_gauss 0

#Error rate: 
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 2 --d_gauss 1

#Error rate: 0.3171
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 2 --d_gauss 2

#Error rate: 0.2733
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 2 --d_gauss 4

#---------------------------------------method 1--------------------------------------#
#Error rate: 0.2021
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 0

#Error rate: 0.5479
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 1

#Error rate: 0.6381
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 2

#Error rate: 0.6915
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 4

#Error rate: 0.7853
#./hw2.py --infile_train_label ./sample/train-labels-idx1-ubyte.gz  --infile_train_image ./sample/train-images-idx3-ubyte.gz --infile_test_label ./sample/t10k-labels-idx1-ubyte.gz --infile_test_image ./sample/t10k-images-idx3-ubyte.gz --toggle 1 -isd 0 --pseudo_cnt_method 1 --PSEUDO_CNST 2300 --method 1 --d_gauss 8

