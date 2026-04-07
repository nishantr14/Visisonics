import urllib.request
import urllib.error
import time
import os
import sys

# Windows Color Hack
os.system('color 0a')

API_BASE = "http://127.0.0.1:7090/simulate/"

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("==========================================================")
    print("      UNAUTHORIZED ROOT ACCESS - OVERRIDE CONSOLE      ")
    print("              TARGET: OT_SHIELD_PROD                   ")
    print("==========================================================")
    print()

def trigger_payload(mode, description):
    print(f"\n[!] INITIATING PAYLOAD: {description}")
    print("[*] establishing tunnel to target node...")
    time.sleep(0.5)
    print("[*] bypassing firewall packet filters...")
    time.sleep(0.5)
    
    url = f"{API_BASE}{mode}"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print(f"[+] SUCCESS! Control overwritten. Payload '{mode}' active on target.")
            else:
                print(f"[-] FAILED. Target responded with status: {response.status}")
    except Exception as e:
        print(f"[-] ERROR DETECTED: Could not reach target node at {url}")
        print(f"    Details: {e}")
    
    print("\nPress ENTER to return to menu...")
    input()

def main():
    while True:
        clear()
        print_header()
        print("Select System Override Vector:\n")
        print("  [1] RESTORE BASELINE (Normalize System)")
        print("  [2] INJECT SQL/NETWORK PAYLOAD (Cyber Attack)")
        print("  [3] SPOOF PLC TELEMETRY (Physical Fault)")
        print("  [4] TRIGGER FULL SYSTEM CHAOS (Quad-Layer Critical)")
        print("  [0] DISCONNECT / EXIT")
        print()
        
        choice = input("root@kali:~# Enter Choice > ").strip()
        
        if choice == '1':
            trigger_payload('normal', 'Restoring standard baseline metrics')
        elif choice == '2':
            trigger_payload('cyber_attack', 'Injecting malicious network packets')
        elif choice == '3':
            trigger_payload('physical_fault', 'Overriding PLC Sensor variables')
        elif choice == '4':
            trigger_payload('critical', 'Executing zero-day Quad-Layer strike')
        elif choice == '0':
            print("\nConnection Terminated.")
            sys.exit(0)
        else:
            print("\n[-] Invalid command sequence.")
            time.sleep(1)

if __name__ == "__main__":
    main()
