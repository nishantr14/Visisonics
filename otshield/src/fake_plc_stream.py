"""
Fake PLC + Network data stream for demo/testing without real hardware.

Provides two independent streams:
  - Physical: BATADAL sensor telemetry (43 columns)
  - Cyber:    TON_IoT network flows (44 columns)
"""

import os
import random
import math
import time
import pandas as pd


class FakePLCStream:
    """Streams BATADAL sensor data (all 43 columns) for physical layer."""

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
        t = time.time()
        reading = {}
        for i in range(1, 8):
            base = 5.0 + 0.5 * math.sin(t * 0.05 + i)
            reading[f'L_T{i}'] = round(base + random.gauss(0, 0.1), 4)
        for i in range(1, 12):
            base = 2.0 + 0.3 * math.sin(t * 0.1 + i) if i <= 4 else 0.0
            reading[f'F_PU{i}'] = round(max(0, base + random.gauss(0, 0.05)), 4)
        reading['F_V2'] = round(1.5 + 0.2 * math.sin(t * 0.08) + random.gauss(0, 0.03), 4)
        for i in range(1, 12):
            reading[f'S_PU{i}'] = 1 if reading.get(f'F_PU{i}', 0) > 0.5 else 0
        reading['S_V2'] = 1 if reading['F_V2'] > 0.5 else 0
        for jname in ['P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                       'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']:
            base = 40.0 + 5.0 * math.sin(t * 0.07 + hash(jname) % 10)
            reading[jname] = round(base + random.gauss(0, 1.0), 4)
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
            reading = {}
            for col in self._sensor_cols:
                if col in row.index:
                    reading[col] = float(row[col])
        else:
            reading = self._synthetic_reading(attack)
            self._index += 1

        if self._mode == "physical_fault":
            for key in [c for c in reading if c.startswith('L_T')]:
                reading[key] = round(reading[key] * 1.3, 4)
        elif self._mode == "critical":
            for key in [c for c in reading if c.startswith(('F_PU', 'P_J', 'L_T', 'F_V'))]:
                reading[key] = round(reading[key] * 1.5, 4)
        return reading


class FakeNetworkStream:
    """Streams TON_IoT network flow data for cyber layer."""

    def __init__(self):
        self._mode = "normal"
        self._index = 0
        self._data = None
        self._normal_rows = None
        self._attack_rows = None
        self._load_csv()

    def _load_csv(self):
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'ton',
            'Train_Test_datasets', 'Train_Test_Network_dataset',
            'train_test_network.csv'
        )
        try:
            df = pd.read_csv(csv_path)
            self._data = df
            self._flow_cols = [c for c in df.columns if c not in ['label', 'type']]
            self._normal_rows = df[df['label'] == 0]
            self._attack_rows = df[df['label'] == 1]
            # Group attacks by type for mode-specific scenarios
            self._attack_types = {}
            for atype in df[df['label'] == 1]['type'].unique():
                self._attack_types[atype] = df[df['type'] == atype]
        except FileNotFoundError:
            self._data = None
            self._flow_cols = []

    def _synthetic_flow(self, attack=False):
        """Fallback: generate synthetic network flow."""
        flow = {
            "src_ip": "192.168.1.100", "dst_ip": "192.168.1.1",
            "src_port": random.randint(1024, 65535),
            "dst_port": random.choice([80, 443, 502, 22, 8080]),
            "proto": random.choice(["tcp", "udp"]),
            "service": random.choice(["http", "dns", "ssl", "-"]),
            "duration": round(random.expovariate(0.1), 6),
            "src_bytes": random.randint(40, 5000),
            "dst_bytes": random.randint(40, 5000),
            "conn_state": random.choice(["SF", "S0", "REJ", "RSTO"]),
            "missed_bytes": 0,
            "src_pkts": random.randint(1, 50),
            "dst_pkts": random.randint(1, 50),
            "src_ip_bytes": random.randint(40, 6000),
            "dst_ip_bytes": random.randint(40, 6000),
            "dns_query": "-", "dns_qclass": 0, "dns_qtype": 0, "dns_rcode": 0,
            "dns_AA": "-", "dns_RD": "-", "dns_RA": "-", "dns_rejected": "-",
            "ssl_version": "-", "ssl_cipher": "-", "ssl_resumed": "-",
            "ssl_established": "-", "ssl_subject": "-", "ssl_issuer": "-",
            "http_trans_depth": "-", "http_method": "-", "http_uri": "-",
            "http_version": "-", "http_request_body_len": 0,
            "http_response_body_len": 0, "http_status_code": 0,
            "http_user_agent": "-", "http_orig_mime_types": "-",
            "http_resp_mime_types": "-",
            "weird_name": "-", "weird_addl": "-", "weird_notice": "-",
        }
        if attack:
            flow["src_bytes"] = random.randint(10000, 100000)
            flow["dst_port"] = 4444
            flow["duration"] = round(random.uniform(100, 500), 6)
        return flow

    def set_mode(self, mode: str):
        self._mode = mode

    def next_reading(self) -> dict:
        if self._data is None:
            attack = self._mode != "normal"
            return self._synthetic_flow(attack)

        # Pick rows based on mode
        if self._mode == "normal":
            subset = self._normal_rows
        elif self._mode == "cyber_attack":
            # Mix of attack types for cyber scenario
            subset = self._attack_rows
        elif self._mode == "physical_fault":
            # Physical fault = normal network traffic (attack is on sensor side)
            subset = self._normal_rows
        elif self._mode == "critical":
            # Critical = attacks on both layers
            subset = self._attack_rows
        else:
            subset = self._normal_rows

        if len(subset) == 0:
            return self._synthetic_flow(self._mode != "normal")

        row = subset.iloc[self._index % len(subset)]
        self._index += 1

        flow = {}
        for col in self._flow_cols:
            val = row[col]
            flow[col] = val if not pd.isna(val) else "-"
        return flow


# Module-level instances
_plc_stream = FakePLCStream()
_net_stream = FakeNetworkStream()


def get_physical_reading() -> dict:
    """Get next BATADAL sensor reading for physical layer."""
    return _plc_stream.next_reading()


def get_cyber_reading() -> dict:
    """Get next TON_IoT network flow for cyber layer."""
    return _net_stream.next_reading()


# Backward compatibility
def get_current_reading() -> dict:
    return get_physical_reading()


def set_mode(mode: str):
    _plc_stream.set_mode(mode)
    _net_stream.set_mode(mode)
