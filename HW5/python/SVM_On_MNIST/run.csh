#! /bin/csh -f
./hw5.py -input_train_x ./data/X_train.csv -input_train_y ./data/Y_train.csv -input_test_x ./data/X_test.csv -input_test_y ./data/Y_test.csv -task 1 -isd 0

./hw5.py -input_train_x ./data/X_train.csv -input_train_y ./data/Y_train.csv -input_test_x ./data/X_test.csv -input_test_y ./data/Y_test.csv -task 2 -search_lin 1 -kfold 5 -isd 0

./hw5.py -input_train_x ./data/X_train.csv -input_train_y ./data/Y_train.csv -input_test_x ./data/X_test.csv -input_test_y ./data/Y_test.csv -task 2 -search_pol 1 -kfold 5 -isd 0

./hw5.py -input_train_x ./data/X_train.csv -input_train_y ./data/Y_train.csv -input_test_x ./data/X_test.csv -input_test_y ./data/Y_test.csv -task 2 -search_rbf 1 -kfold 5 -isd 0

./hw5.py -input_train_x ./data/X_train.csv -input_train_y ./data/Y_train.csv -input_test_x ./data/X_test.csv -input_test_y ./data/Y_test.csv -task 3 -kfold 5 -isd 0
