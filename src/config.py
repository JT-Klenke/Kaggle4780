import torch

BATCH_SIZE = 512

DATA_DIR = "data/"
OUT_DIR = "out/"

RDB_PATH = f"sqlite:///{OUT_DIR}cs4780.db"
STUDY_NAME = "cs4780"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
