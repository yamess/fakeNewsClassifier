import torch
from transformers import BertTokenizer

SEED = 42
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 2
DROPOUT = 0.3
LEARNING_RATE = 4e-5
TRAINING_STAT_PATH = "output/"
DEVICE = torch.device("cuda")
# 'cuda' if torch.cuda.is_available() else 'cpu'
BERT_TOKENIZER_PATH = "bert-base-uncased"
BERT_MODEL_PATH = "bert-base-uncased"

DATA_PATH = "data/"

LOG_DIR = "output/log/"
OUTPUT_DIR = "output/"
CHECKPOINT = "output/checkpoint.pt"
OUTPUT_MODEL_FILE = "model.bin"
