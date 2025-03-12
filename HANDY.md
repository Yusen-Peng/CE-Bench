# handy commands

# GPU usage
salloc --account=PAS2912 --nodes=1 --ntasks-per-node=1 --gres=gpu:v100:1 --time=0:20:00
nvidia-smi
nvidia-smi -L
squeue -j ####### -o "%i %T %r"

# run experiments
python3 -m preliminary_exploration.L0_test

# enviroment setup
module load miniconda3
conda create --name kan_llama python=3.10 (only once)
conda conda install numpy pandas scipy matplotlib (only once)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (only once)
conda activate kan_llama


