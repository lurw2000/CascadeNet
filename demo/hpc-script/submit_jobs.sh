
sbatch --export=path=config/cascadegan_test/config-ton-vanilla_LM-5_200.json demo/hpc-script/run-cascadegan_test.slurm
sbatch --export=path=config/cascadegan_test/config-ton-vanilla_LM-5_200-mlp_packetrate.json demo/hpc-script/run-cascadegan_test.slurm
sbatch --export=path=config/cascadegan_test/config-ton-vanilla_LM-5_200-lstm_packetfield.json demo/hpc-script/run-cascadegan_test.slurm
sbatch --export=path=config/cascadegan_test/config-ton-vanilla_LM-5_200-mlp_packetrate-lstm_packetfield.json demo/hpc-script/run-cascadegan_test.slurm

# sbatch --export=path=/gpfsnyu/home/rl4689/ML-testing-dev/config/cascadegan_sigcomm24/config-ton-vanilla_LM-5_200.json run-cascadegan_sigcomm24.slurm
# sbatch --export=path=/gpfsnyu/home/rl4689/ML-testing-dev/config/cascadegan_sigcomm24/config-caida-vanilla_LM-5_200.json run-cascadegan_sigcomm24.slurm
# sbatch --export=path=/gpfsnyu/home/rl4689/ML-testing-dev/config/cascadegan_sigcomm24/config-dc-vanilla_LM-5_200.json run-cascadegan_sigcomm24.slurm
# sbatch --export=path=/gpfsnyu/home/rl4689/ML-testing-dev/config/cascadegan_sigcomm24/config-ca-vanilla_LM-5_200.json run-cascadegan_sigcomm24.slurm
