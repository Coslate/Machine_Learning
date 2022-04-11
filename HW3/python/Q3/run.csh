#! /bin/csh -f

rm -rf ./ans*.log

#./hw3.py -b 1   -n 4 -a 1 -w '[1, 2, 3, 4]' --gaussian_meth 0 -isd 0 -eps 0.0000000001 > ans1.log
#./hw3.py -b 100 -n 4 -a 1 -w '[1, 2, 3, 4]' --gaussian_meth 0 -isd 0 -eps 0.0000000001 > ans2.log
#./hw3.py -b 1   -n 3 -a 3 -w '[1, 2, 3]'    --gaussian_meth 0 -isd 0 -eps 0.0000000001 > ans3.log

./hw3.py -b 1   -n 4 -a 1 -w '[1, 2, 3, 4]' --gaussian_meth 1 -isd 0 -eps 0.00000001   > ans1.log
./hw3.py -b 100 -n 4 -a 1 -w '[1, 2, 3, 4]' --gaussian_meth 1 -isd 0 -eps 0.00000001   > ans2.log
./hw3.py -b 1   -n 3 -a 3 -w '[1, 2, 3]'    --gaussian_meth 1 -isd 0 -eps 0.00000001   > ans3.log
#./hw3.py -b 100 -n 4 -a 1 -w '[1, 2, 3, 4]' --gaussian_meth 1 -isd 0 -eps 0.000001     > ans2.log
