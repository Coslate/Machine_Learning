#! /bin/csh -f

set experiment_s    = 1
set experiment_t    = 5
set cluster_num     = 7
@ i = $experiment_s
while ($i <= $experiment_t)
    echo "experiment_t = $i"

    @ j = 2
    while ($j <= $cluster_num)
    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523/0523_${i}/output_2022_0523_c${j}/init_m0"
    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 0 -gfk 8 -isd 1 -dir $out_folder_name

        @ j += 1
    end

    @ j = 2
    while ($j <= $cluster_num)
    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523/0523_${i}/output_2022_0523_c${j}/init_m1"
    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 1 -gfk 8 -isd 1 -dir $out_folder_name

        @ j += 1
    end

    @ i += 1
end

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 2 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c2/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 3 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c3/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 4 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c4/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 5 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c5/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 6 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c6/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 7 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c7/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 8 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c8/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 9 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c9/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 10 -init_m 0 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c10/init_m0'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 2 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c2/init_m1'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 3 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c3/init_m1'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 4 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c4/init_m1'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 5 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c5/init_m1'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 6 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c6/init_m1'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 7 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c7/init_m1'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 8 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c8/init_m1'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 9 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c9/init_m1'

#./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 10 -init_m 1 -gfk 8 -isd 1 -dir '/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0523_3/output_2022_0523_3_c10/init_m1'
