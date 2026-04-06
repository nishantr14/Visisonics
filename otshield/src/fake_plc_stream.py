"""Fake PLC data stream for demo/testing without real hardware."""

import os
import random
import math
import time
import pandas as pd


class FakePLCStream:
    """Streams BATADAL sensor data (all 43 columns) for testing."""

    def __init__(self):
        self._mode = "normal"
        self._index = 0
        self._data = None
        self._normal_rows = None
        self._attack_rows = None
        self._load_csv()

    def _load_csv(self):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'BATADAL_dataset04.csv')
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if 'ATT_FLAG' in df.columns:
                self._data = df
                self._sensor_cols = [c for c in df.columns if c not in ['DATETIME', 'ATT_FLAG']]
                self._normal_rows = df[df['ATT_FLAG'] != 1]
                self._attack_rows = df[df['ATT_FLAG'] == 1]
        except FileNotFoundError:
            self._data = None
            self._sensor_cols = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
                                 'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3',
                                 'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6',
                                 'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9',
                                 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2',
                                 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                                 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

    def _synthetic_reading(self, attack=False):
        """Fallback: generate synthetic data for all 43 sensors."""
        t = time.time()
        reading = {}
        # Levels
        for i in range(1, 8):
            base = 5.0 + 0.5 * math.sin(t * 0.05 + i)
            reading[f'L_T{i}'] = round(base + random.gauss(0, 0.1), 4)
        # Flows
        for i in range(1, 12):
            base = 2.0 + 0.3 * math.sin(t * 0.1 + i) if i <= 4 else 0.0
            reading[f'F_PU{i}'] = round(max(0, base + random.gauss(0, 0.05)), 4)
        reading['F_V2'] = round(1.5 + 0.2 * math.sin(t * 0.08) + random.gauss(0, 0.03), 4)
        # Statuses
        for i in range(1, 12):
            reading[f'S_PU{i}'] = 1 if reading.get(f'F_PU{i}', 0) > 0.5 else 0
        reading['S_V2'] = 1 if reading['F_V2'] > 0.5 else 0
        # Pressures
        for jname in ['P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                       'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']:
            base = 40.0 + 5.0 * math.sin(t * 0.07 + hash(jname) % 10)
            reading[jname] = round(base + random.gauss(0, 1.0), 4)

        reading['ATT_FLAG'] = 1 if attack else 0

        if attack:
            reading['F_PU1'] += random.uniform(0.5, 2.0)
            reading['P_J280'] += random.uniform(5, 15)
            reading['L_T1'] += random.uniform(1.0, 3.0)

        return reading

    def set_mode(self, mode: str):
        valid = ("normal", "cyber_attack", "physical_fault", "critical")
        if mode not in valid:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid}")
        self._mode = mode

    def next_reading(self) -> dict:
        attack = self._mode != "normal"

        if self._data is not None:
            subset = self._attack_rows if attack else self._normal_rows
            if len(subset) == 0:
                return self._synthetic_reading(attack)

            row = subset.iloc[self._index % len(subset)]
            self._index += 1

            # Return ALL sensor columns (not just 9)
            reading = {}
            for col in self._sensor_cols:
                if col in row.index:
                    reading[col] = float(row[col])
            reading['ATT_FLAG'] = int(row.get('ATT_FLAG', 0))
        else:
            reading = self._synthetic_reading(attack)
            self._index += 1

        # Apply mode-specific amplification
        if self._mode == "physical_fault":
            for key in [c for c in reading if c.startswith('L_T')]:
                reading[key] = round(reading[key] * 1.3, 4)
        elif self._mode == "critical":
            for key in [c for c in reading if c.startswith(('F_PU', 'P_J', 'L_T', 'F_V'))]:
                reading[key] = round(reading[key] * 1.5, 4)

        return reading


# Module-level instance for api.py import
_stream = FakePLCStream()


def get_current_reading() -> dict:
    return _stream.next_reading()


def set_mode(mode: str):
    _stream.set_mode(mode)
