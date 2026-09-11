import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IMPUTATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(IMPUTATION_DIR)

seq_len = "50"
model = "ARI-LLM"
percent = "100"
mask_rate = "0.1"
train_epochs = "1"
sample_num = "3000"
llm_model = "gpt2"
Lambda = "4"
itr = "1"
features = "300"

command = [
    os.environ.get("PYTHON", sys.executable), os.path.join(IMPUTATION_DIR, "run.py"),
    "--train_epochs", train_epochs,
    "--itr", itr,
    "--task_name", "imputation",
    "--is_training", "1",
    "--root_path", os.path.join(PROJECT_DIR, "datasets", "net_traffic", "GEANT"),
    "--data_path", "geant.csv",
    "--model_id", f"geant-feature_{features}_{model}_samplenum{sample_num}",
    "--sample_num", sample_num,
    "--llm_model", llm_model,
    "--data", "net_traffic_geant",
    "--seq_len", seq_len,
    "--batch_size", "35",
    "--learning_rate", "0.001",
    "--mlp", "1",
    "--d_model", "768",
    "--n_heads", "4",
    "--d_ff", "768",
    "--enc_in", seq_len,
    "--dec_in", features,
    "--c_out", features,
    "--freq", "h",
    "--Lambda", Lambda,
    "--percent", percent,
    "--gpt_layers", "6",
    "--model", model,
    "--patience", "5",
    "--mask_rate", mask_rate,
]

subprocess.run(command, check=True)
