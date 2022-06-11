#! /bin/csh -f

set experiment_s    = 1
set experiment_t    = 1
set perplex_list  = (10 30 50 100 200 300)

#clean all

@ i = $experiment_s
while ($i <= $experiment_t)
    echo "experiment_t = $i"

    foreach perplex ($perplex_list)
        set output_dir =  "/home/bcc/File_Change/Machine_Learning/HW7/Output_SNE/Symmetric-SNE_perplexity_${perplex}"
        #clean all
        rm -rf $output_dir/*
        ./hw7.py -img ./tsne_python/mnist2500_X.txt -ilb ./tsne_python/mnist2500_labels.txt -mode 0 -perp ${perplex} -et 0 -et_eps 1e-8 -dir ${output_dir} -isd 1

        set output_dir =  "/home/bcc/File_Change/Machine_Learning/HW7/Output_SNE/t-SNE_perplexity_${perplex}"
        #clean all
        rm -rf $output_dir/*
        ./hw7.py -img ./tsne_python/mnist2500_X.txt -ilb ./tsne_python/mnist2500_labels.txt -mode 1 -perp ${perplex} -et 0 -et_eps 1e-8 -dir ${output_dir} -isd 1
    end

    @ i += 1
end
