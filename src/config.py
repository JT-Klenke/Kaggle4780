import torch

DATA_DIR = "data/"
OUT_DIR = "out/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
