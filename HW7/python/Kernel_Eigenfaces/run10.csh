#! /bin/csh -f

#./hw7.py -itrd ./Yale_Face_Database/Training -ited ./Yale_Face_Database/Testing -isd 1 -lmk_pca 25 -lmk_lda 25 -knn 5 -row 50 -col 50 -auc 0 -rg 0.000001 -odr "/home/bcc/File_Change/Machine_Learning/HW7/Output_Kernel_Eigenfaces"
set experiment_s    = 1
set experiment_t    = 1
set pol_g_list  = (0.00000003 0.00390625 0.0004 0.000001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 300000)
set pol_c0_list = (265 5 0)
set pol_d_list  = (0 1 3 5 7 9 11)
set rbf_g_list  = (0.000000001 0.00000003 0.000001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 30000000)
set k_list      = (10)

#clean all

@ i = $experiment_s
while ($i <= $experiment_t)
    echo "experiment_t = $i"

    foreach k ($k_list)
        set output_dir =  "/home/bcc/File_Change/Machine_Learning/HW7/Output/Output_Eigenfaces_Kernel_GridSearch_KNN_${k}"
        #clean all
        rm -rf $output_dir/*
        foreach rbf_g ($rbf_g_list)
            foreach pol_g ($pol_g_list)
                foreach pol_c0 ($pol_c0_list)
                    foreach pol_d ($pol_d_list)
                        echo "k = ${k}, rbf_g = $rbf_g, pol_g = ${pol_g}, pol_c0 = ${pol_c0}, pol_d = ${pol_d}"
                        ./hw7.py -itrd ./Yale_Face_Database/Training -ited ./Yale_Face_Database/Testing -isd 0 -lmk_pca 25 -lmk_lda 25 -knn ${k} -row 50 -col 50 -auc 0 -km 1 -pg ${pol_g} -pc ${pol_c0} -pd ${pol_d} -rg ${rbf_g} -odr ${output_dir}
                    end
                end
            end
        end

        set output_dir =  "/home/bcc/File_Change/Machine_Learning/HW7/Output/Output_Eigenfaces_KNN_${k}"
        #clean all
        rm -rf $output_dir/*
        ./hw7.py -itrd ./Yale_Face_Database/Training -ited ./Yale_Face_Database/Testing -isd 0 -lmk_pca 25 -lmk_lda 25 -knn ${k} -row 50 -col 50 -auc 0 -km 0 -odr ${output_dir}
    end

    @ i += 1
end
