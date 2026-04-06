"""Real OpenPLC Modbus data collector for OTShield.

Reads holding registers from an OpenPLC runtime via Modbus TCP.
Maps register addresses to BATADAL sensor names.

NOTE: The register map below assumes OpenPLC is configured with
the BATADAL water treatment ladder logic. Adjust addresses to match
your actual PLC program.
"""

from pymodbus.client import ModbusTcpClient

MODBUS_HOST = "localhost"
MODBUS_PORT = 502

# Map BATADAL sensor names -> (register_address, scale_divisor)
# Registers without a divisor (None) are read as raw integers (status flags).
# Adjust these register addresses to match your OpenPLC program.
REGISTER_MAP = {
    # Tank levels
    'L_T1':    (0, 100.0),
    'L_T2':    (1, 100.0),
    'L_T3':    (2, 100.0),
    'L_T4':    (3, 100.0),
    'L_T5':    (4, 100.0),
    'L_T6':    (5, 100.0),
    'L_T7':    (6, 100.0),
    # Pump flows
    'F_PU1':   (7, 100.0),
    'F_PU2':   (8, 100.0),
    'F_PU3':   (9, 100.0),
    'F_PU4':   (10, 100.0),
    'F_PU5':   (11, 100.0),
    'F_PU6':   (12, 100.0),
    'F_PU7':   (13, 100.0),
    'F_PU8':   (14, 100.0),
    'F_PU9':   (15, 100.0),
    'F_PU10':  (16, 100.0),
    'F_PU11':  (17, 100.0),
    'F_V2':    (18, 100.0),
    # Pump statuses (binary 0/1)
    'S_PU1':   (19, None),
    'S_PU2':   (20, None),
    'S_PU3':   (21, None),
    'S_PU4':   (22, None),
    'S_PU5':   (23, None),
    'S_PU6':   (24, None),
    'S_PU7':   (25, None),
    'S_PU8':   (26, None),
    'S_PU9':   (27, None),
    'S_PU10':  (28, None),
    'S_PU11':  (29, None),
    'S_V2':    (30, None),
    # Junction pressures
    'P_J280':  (31, 100.0),
    'P_J269':  (32, 100.0),
    'P_J300':  (33, 100.0),
    'P_J256':  (34, 100.0),
    'P_J289':  (35, 100.0),
    'P_J415':  (36, 100.0),
    'P_J302':  (37, 100.0),
    'P_J306':  (38, 100.0),
    'P_J307':  (39, 100.0),
    'P_J317':  (40, 100.0),
    'P_J14':   (41, 100.0),
    'P_J422':  (42, 100.0),
}

NUM_REGISTERS = max(addr for addr, _ in REGISTER_MAP.values()) + 1


def get_current_reading() -> dict:
    """Read all holding registers from OpenPLC via Modbus TCP."""
    client = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
    if not client.connect():
        raise RuntimeError(
            f"Cannot connect to OpenPLC at {MODBUS_HOST}:{MODBUS_PORT}. "
            "Ensure OpenPLC is running and Modbus TCP is enabled on port 502."
        )

    try:
        result = client.read_holding_registers(0, NUM_REGISTERS)
        if result.isError():
            raise RuntimeError(f"Modbus read error: {result}")

        regs = result.registers
        reading = {}
        for name, (idx, divisor) in REGISTER_MAP.items():
            if divisor is not None:
                reading[name] = round(regs[idx] / divisor, 4)
            else:
                reading[name] = int(regs[idx])

        reading['ATT_FLAG'] = 0  # always 0 for live PLC data
        return reading
    finally:
        client.close()
