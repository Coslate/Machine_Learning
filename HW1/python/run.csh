#! /bin/csh -f

#./hw1.py --input_file testfile.txt --poly_num 2 --lamb 200 -isd 1
./hw1.py --input_file testfile.txt --poly_num 2 --lamb  0 > ./ans.log
./hw1.py --input_file testfile.txt --poly_num 3 --lamb  0 >> ./ans.log
./hw1.py --input_file testfile.txt --poly_num 3 --lamb  10000 >> ./ans.log

