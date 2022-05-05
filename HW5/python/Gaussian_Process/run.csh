#! /bin/csh -f

./hw5.py --input_file ./data/input.data -alpha 1 -ls 1 -beta 5 -test_num 3000 -opt_hyper 0 -isd 0

./hw5.py --input_file ./data/input.data -alpha 1 -ls 1 -beta 5 -test_num 3000 -opt_hyper 1 -isd 0

./hw5.py --input_file ./data/input.data -alpha 1 -ls 1 -beta 5 -test_num 3000 -opt_hyper 0 -sigma 2 -use_sigma 1 -isd 0

./hw5.py --input_file ./data/input.data -alpha 1 -ls 1 -beta 5 -test_num 3000 -opt_hyper 1 -sigma 2 -use_sigma 1 -isd 0

