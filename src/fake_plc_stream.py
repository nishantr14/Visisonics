"""Fake PLC data stream for demo/testing without real hardware."""

import os
import random
import math
import time
import pandas as pd


class FakePLCStream:
    COLUMNS = ['F_PU1', 'F_PU2', 'P_J280', 'P_J269', 'L_T1', 'L_T2', 'L_T3', 'S_PU1', 'ATT_FLAG']

    def __init__(self):
        self._mode = "normal"
        self._index = 0
        self._data = None
        self._load_csv()

    def _load_csv(self):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'BATADAL_dataset04.csv')
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names
            df.columns = [c.strip() for c in df.columns]
            if 'ATT_FLAG' in df.columns:
                self._data = df
        except FileNotFoundError:
            self._data = None

    def _synthetic_reading(self, attack=False):
        t = time.time()
        base = {
            'F_PU1':  round(2.0 + 0.3 * math.sin(t * 0.1) + random.gauss(0, 0.05), 4),
            'F_PU2':  round(1.8 + 0.2 * math.sin(t * 0.15) + random.gauss(0, 0.04), 4),
            'P_J280': round(0.5 + 0.1 * math.sin(t * 0.08) + random.gauss(0, 0.02), 4),
            'P_J269': round(0.45 + 0.08 * math.sin(t * 0.12) + random.gauss(0, 0.02), 4),
            'L_T1':   round(5.0 + 0.5 * math.sin(t * 0.05) + random.gauss(0, 0.1), 4),
            'L_T2':   round(4.5 + 0.4 * math.sin(t * 0.06) + random.gauss(0, 0.1), 4),
            'L_T3':   round(4.8 + 0.3 * math.sin(t * 0.07) + random.gauss(0, 0.1), 4),
            'S_PU1':  1,
            'ATT_FLAG': 1 if attack else 0,
        }
        if attack:
            base['F_PU1'] += random.uniform(0.5, 2.0)
            base['P_J280'] += random.uniform(0.3, 1.0)
            base['L_T1'] += random.uniform(1.0, 3.0)
        return base

    def set_mode(self, mode: str):
        valid = ("normal", "cyber_attack", "physical_fault", "critical")
        if mode not in valid:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid}")
        self._mode = mode

    def next_reading(self) -> dict:
        attack = self._mode != "normal"

        if self._data is not None:
            if attack:
                subset = self._data[self._data['ATT_FLAG'] == 1]
            else:
                subset = self._data[self._data['ATT_FLAG'] == 0]

            if len(subset) == 0:
                return self._synthetic_reading(attack)

            row = subset.iloc[self._index % len(subset)]
            self._index += 1
            reading = {col: float(row[col]) if col != 'ATT_FLAG' else int(row[col])
                       for col in self.COLUMNS if col in row.index}
        else:
            reading = self._synthetic_reading(attack)
            self._index += 1

        # Apply mode-specific amplification
        if self._mode == "physical_fault":
            for key in ('L_T1', 'L_T2', 'L_T3'):
                if key in reading:
                    reading[key] = round(reading[key] * 1.3, 4)
        elif self._mode == "critical":
            for key in ('F_PU1', 'F_PU2', 'P_J280', 'P_J269', 'L_T1', 'L_T2', 'L_T3'):
                if key in reading:
                    reading[key] = round(reading[key] * 1.5, 4)

        return reading


# Module-level instance for api.py import
_stream = FakePLCStream()


def get_current_reading() -> dict:
    return _stream.next_reading()


def set_mode(mode: str):
    _stream.set_mode(mode)
