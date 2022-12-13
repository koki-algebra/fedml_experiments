from typing import Tuple
from pathlib import Path

import pandas as pd
import fedml
from fedml.arguments import Arguments


def load_uci_income(args: Arguments) -> Tuple[list, int]:
    parent = Path(__file__).resolve().parent
    file_path = parent.joinpath("data/uci_income/adult.csv")
    df = pd.read_csv(file_path)
