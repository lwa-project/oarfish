import os
import glob

from typing import List

MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

def get_all_models() -> List[str]:
    models = glob.glob(os.path.join(MODEL_PATH, '*.pt'))
    models.sort()
    return models


def get_default_binary_model() -> str:
    return os.path.join(MODEL_PATH, 'binary.pt')


def get_default_multi_model() -> str:
    return os.path.join(MODEL_PATH, 'multi.pt')
