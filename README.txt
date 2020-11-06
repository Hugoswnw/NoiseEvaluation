BENCHMARK IMPLEMENTATION FOR "Comparison of Weakly Supervised Learning Methods on the Case of Fraud Detection"

Description of folders :
 - Benchmark : Sources for the module implemented
 - Cleaned_Datasets : Datasets cleaned and standardized with STEP 1
 - Datasets : Raw datasets collected from public repositories
 - NoisedKFold : Parent of folders that stores intermediate split and noised datasets at the end of STEP 2. Name of the folder corresponds to "RUN_{R}_{K}_{|RHO|}" where R, K and RHO are protocol parameters.
 - Notebooks : Contains formerly used notebooks. Originally used for exploratory works and tests, only here for legacy/backup purposes.
 - pykhiops : source folder of the Khiops python linker. Can be extracted from Khiops installation.
 - Results : Folder where results of STEP 3 are stored. Name of each file corresponds to "results_{dataset name}_{R}_{K}_{|RHO|}" where R, K and RHO are protocol parameters.
 - Results_img : Folder where plots generated in STEP 4 are stored. Mostly for the report writing.



To run the protocol described in the report ( standardization -> split -> noise addition -> model training -> results analysis )
simply follow the notebooks order.
