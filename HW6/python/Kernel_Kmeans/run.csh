#! /bin/csh -f

set experiment_s    = 1
set experiment_t    = 1
set cluster_num     = 4
set gs_list = (0.0001 0.001 0.01 0.1 1 10 100 1000 10000)
set gc_list = (0.0001 0.001 0.01 0.1 1 10 100 1000 10000)

@ i = $experiment_s
while ($i <= $experiment_t)
    echo "experiment_t = $i"

    @ j = 2
    while ($j <= $cluster_num)
        foreach gs ($gs_list)
            foreach gc ($gc_list)
                set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0526/0526_${i}/gs_${gs}_gc_${gc}/output_2022_0526_c${j}/init_m0"
                ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs ${gs} -gc ${gc} -cn ${j} -init_m 0 -gfk 8 -isd 1 -dir $out_folder_name
            end
        end

        @ j += 1
    end

    @ j = 2
    while ($j <= $cluster_num)
        foreach gs ($gs_list)
            foreach gc ($gc_list)
                set out_folder_name = "/home/bcc/File_Change/Machine_Learning/HW6/Output_KernelKmeans_Clustering/0526/0526_${i}/gs_${gs}_gc_${gc}/output_2022_0526_c${j}/init_m1"
                ./hw6.py -img1 ./data/image1.png -img2 ./data/image2.png -gs ${gs} -gc ${gc} -cn ${j} -init_m 1 -gfk 8 -isd 1 -dir $out_folder_name
            end
        end

        @ j += 1
    end

    @ i += 1
end

