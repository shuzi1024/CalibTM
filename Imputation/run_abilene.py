import os
import subprocess
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IMPUTATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(IMPUTATION_DIR)

seq_len = "50"
model = "ARI-LLM"
percent = "100"
mask_rate = "0.1"
train_epochs = "1"
sample_num = "500"
llm_model = "gpt2"
Lambda = "2"
itr = "1"
features = "144"

command = [
    os.environ.get("PYTHON", sys.executable), os.path.join(IMPUTATION_DIR, "run.py"),
    "--train_epochs", train_epochs,
    "--itr", itr,
    "--task_name", "imputation",
    "--is_training", "1",
    "--root_path", os.path.join(PROJECT_DIR, "datasets", "net_traffic", "Abilene"),
    "--data_path", "abilene.csv",
    "--real_data_path", "abilene.csv",
    "--model_id", f"Abilene-few-shot-rate{mask_rate}_{model}_samplenum{sample_num}_seq_{seq_len}",
    "--sample_num", sample_num,
    "--llm_model", llm_model,
    "--data", "net_traffic_abilene",
    "--seq_len", seq_len,
    "--batch_size", "60",
    "--learning_rate", "0.001",
    "--mlp", "1",
    "--d_model", "768",
    "--n_heads", "4",
    "--d_ff", "768",
    "--enc_in", seq_len,
    "--dec_in", features,
    "--c_out", features,
    "--Lambda", Lambda,
    "--freq", "h",
    "--percent", percent,
    "--gpt_layers", "6",
    "--model", model,
    "--patience", "5",
    "--mask_rate", mask_rate,
]

subprocess.run(command, check=True)
