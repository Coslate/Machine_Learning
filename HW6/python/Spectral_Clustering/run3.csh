#! /bin/csh -f

set experiment_s    = 1
set experiment_t    = 1
set cluster_num     = 4
@ i = $experiment_s
while ($i <= $experiment_t)
    echo "experiment_t = $i"

#    @ j = 2
#    while ($j <= $cluster_num)
#    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d0/0525_${i}/output_2022_0525_c${j}/init_m0_ratio_cut"
#    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 0 -gfk 8 -dis 0 -mod 0 -isd 1 -dir $out_folder_name

#    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d0/0525_${i}/output_2022_0525_c${j}/init_m0_normalized_cut"
#    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 0 -gfk 8 -dis 0 -mod 1 -isd 1 -dir $out_folder_name

#        @ j += 1
#    end

#    @ j = 2
#    while ($j <= $cluster_num)
#    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d0/0525_${i}/output_2022_0525_c${j}/init_m1_ratio_cut"
#    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 1 -gfk 8 -dis 0 -mod 0 -isd 1 -dir $out_folder_name
#    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d0/0525_${i}/output_2022_0525_c${j}/init_m1_normalized_cut"
#    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 1 -gfk 8 -dis 0 -mod 1 -isd 1 -dir $out_folder_name

#        @ j += 1
#    end


    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d0/0525_1/output_2022_0525_c4/init_m1_ratio_cut"
    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn 4 -init_m 1 -gfk 8 -dis 0 -mod 0 -isd 1 -dir $out_folder_name

#    @ j = 2
#    while ($j <= $cluster_num)
#    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d1/0525_${i}/output_2022_0525_c${j}/init_m0_ratio_cut"
#    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 0 -gfk 8 -dis 1 -mod 0 -isd 1 -dir $out_folder_name

#    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d1/0525_${i}/output_2022_0525_c${j}/init_m0_normalized_cut"
#    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 0 -gfk 8 -dis 1 -mod 1 -isd 1 -dir $out_folder_name

#        @ j += 1
#    end

#    @ j = 2
#    while ($j <= $cluster_num)
#    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d1/0525_${i}/output_2022_0525_c${j}/init_m1_ratio_cut"
#    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 1 -gfk 8 -dis 1 -mod 0 -isd 1 -dir $out_folder_name
#    set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_Spectral_Clustering/0525/0525_d1/0525_${i}/output_2022_0525_c${j}/init_m1_normalized_cut"
#    ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs 0.0001 -gc 0.0001 -cn ${j} -init_m 1 -gfk 8 -dis 1 -mod 1 -isd 1 -dir $out_folder_name

#        @ j += 1
#    end

    @ i += 1
end

