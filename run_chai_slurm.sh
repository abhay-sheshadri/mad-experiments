export HF_HOME="/nas/ucb/$USER/hf_cache"
eval "$(/nas/ucb/$USER/anaconda3/bin/conda shell.bash hook)"
conda activate
cd /nas/ucb/$USER/mad-experiments
huggingface-cli login --token "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"
python run_experiments.py
