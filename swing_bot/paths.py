"""Small repository path helpers."""
from __future__ import annotations
from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def data_dir() -> Path:
    return repo_root() / "data"

def raw_data_dir() -> Path:
    return data_dir() / "raw"

def processed_data_dir() -> Path:
    return data_dir() / "processed"

def outputs_dir() -> Path:
    return repo_root() / "outputs"

def models_dir() -> Path:
    return repo_root() / "models"

def btcjpy_1m_raw_dir() -> Path:
    return raw_data_dir() / "binance_japan" / "BTCJPY" / "1m"

def btcjpy_1m_processed_dir() -> Path:
    return processed_data_dir() / "binance_japan" / "BTCJPY" / "1m"
