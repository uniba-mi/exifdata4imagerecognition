import pandas as pd
from pathlib import Path

def readCSVDataFrame(filePath: str, **args) -> pd.DataFrame:
    path = Path(filePath).resolve()
    if not path.exists():
        return None
    return pd.read_csv(path, index_col = 0, **args)

def writeCSVDataFrame(dataFrame: pd.DataFrame, filePath: str, override: bool = False):
    path = Path(filePath).resolve()
    if path.exists() and override == False:
        return
    else:
        path.parent.mkdir(parents = True, exist_ok = True)
        dataFrame.to_csv(path)