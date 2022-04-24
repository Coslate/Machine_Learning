#! /bin/csh -f

./hw4.py -N 50 -mx1 1 -my1 1 -mx2 10 -my2 10 -vx1 2 -vy1 2 -vx2 2 -vy2 2 --epsilon_grad 0.0001 --epsilon_newt 0.0001 --use_my_inverse 0 -gm 1 --learning_rate 0.01 -tcg 10000 -tcn 10000 -isd 0

./hw4.py -N 50 -mx1 1 -my1 1 -mx2 3 -my2 3 -vx1 2 -vy1 2 -vx2 4 -vy2 4 --epsilon_grad 0.000001 --epsilon_newt 0.000001 --use_my_inverse 0 -gm 1 --learning_rate 1 -tcg 200000 -tcn 200000 -isd 0
