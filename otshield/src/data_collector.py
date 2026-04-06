"""Real OpenPLC Modbus data collector for OTShield."""

from pymodbus.client import ModbusTcpClient

MODBUS_HOST = "localhost"
MODBUS_PORT = 502

REGISTER_MAP = {
    'F_PU1':    (0, 100.0),
    'P_J280':   (1, 100.0),
    'L_T1':     (2, 100.0),
    'L_T2':     (3, 100.0),
    'L_T3':     (4, 100.0),
    'F_PU2':    (5, 100.0),
    'P_J269':   (6, 100.0),
    'S_PU1':    (7, None),   # raw int
}


def get_current_reading() -> dict:
    """Read holding registers 0-9 from OpenPLC via Modbus TCP."""
    client = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
    if not client.connect():
        raise RuntimeError(
            f"Cannot connect to OpenPLC at {MODBUS_HOST}:{MODBUS_PORT}. "
            "Ensure OpenPLC is running and Modbus TCP is enabled on port 502."
        )

    try:
        result = client.read_holding_registers(0, 10)
        if result.isError():
            raise RuntimeError(f"Modbus read error: {result}")

        regs = result.registers
        reading = {}
        for name, (idx, divisor) in REGISTER_MAP.items():
            if divisor is not None:
                reading[name] = round(regs[idx] / divisor, 4)
            else:
                reading[name] = int(regs[idx])

        reading['ATT_FLAG'] = 0  # always 0 for live PLC
        return reading
    finally:
        client.close()
