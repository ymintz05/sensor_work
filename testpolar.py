#!/usr/bin/env python3
"""
Polar H10 Stream Sniffer (Bleak-version-safe)
- Finds Polar H10 by POLAR_H10_ID substring in advertised name
- Connects
- Enumerates all characteristics that support notify/indicate
- Subscribes to all of them and prints any output for LISTEN_DURATION seconds
"""

import asyncio
import time
from bleak import BleakScanner, BleakClient

POLAR_H10_ID = "0C83F23F"
SCAN_TIMEOUT = 10
LISTEN_DURATION = 30


def decode_hr_if_applicable(data: bytearray):
    """Best-effort decode of standard Heart Rate Measurement payload (0x2A37)."""
    if not data or len(data) < 2:
        return None
    flags = data[0]
    try:
        if flags & 0x01:  # uint16
            if len(data) >= 3:
                return int.from_bytes(data[1:3], "little")
            return None
        else:            # uint8
            return data[1]
    except Exception:
        return None


def make_notify_callback(uuid: str):
    def cb(sender, data: bytearray):
        ts = time.strftime("%H:%M:%S")
        hex_data = data.hex(" ")
        hr = decode_hr_if_applicable(data)
        if hr is not None:
            print(f"[{ts}] 🔔 {uuid}  ❤️ HR={hr} bpm   raw={hex_data}")
        else:
            print(f"[{ts}] 🔔 {uuid}  raw={hex_data}  (len={len(data)})")
    return cb


async def find_polar_h10(timeout=SCAN_TIMEOUT):
    print(f"Scanning for Polar H10 (ID contains: {POLAR_H10_ID}) for {timeout}s...")
    print("Make sure the H10 is on and not connected to Polar Beat/Flow/another device.\n")

    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)

    # Prefer exact ID substring match in name
    for _, (device, adv) in devices.items():
        if device.name and POLAR_H10_ID in device.name:
            rssi = getattr(adv, "rssi", "N/A")
            print(f"✓ Found your Polar H10: {device.name}")
            print(f"  Address: {device.address}")
            print(f"  RSSI: {rssi} dBm\n")
            return device

    # Fallback: any "polar"
    polar = []
    for _, (device, adv) in devices.items():
        if device.name and "polar" in device.name.lower():
            polar.append((device, adv))

    if not polar:
        print("No Polar devices found.\n")
        return None

    print(f"Found {len(polar)} Polar device(s) (not matching your ID substring):\n")
    for i, (device, adv) in enumerate(polar, start=1):
        rssi = getattr(adv, "rssi", "N/A")
        print(f"{i}. {device.name}  {device.address}  RSSI={rssi} dBm")
    print()

    # If multiple, pick the first (you can add selection if you want)
    return polar[0][0]


async def resolve_services(client: BleakClient):
    """
    Bleak-version-safe service resolution.
    - Many Bleak versions populate client.services after connecting.
    - Some versions have get_services(), some don't.
    """
    # Try get_services if it exists
    if hasattr(client, "get_services"):
        try:
            services = await client.get_services()
            if services:
                return services
        except Exception:
            pass

    # Fallback to client.services
    services = getattr(client, "services", None)
    return services


async def dump_gatt(services):
    print("\nGATT map (services/characteristics):")
    print("=" * 60)
    for service in services:
        print(f"\nSERVICE {service.uuid} ({getattr(service, 'description', '')})")
        for char in service.characteristics:
            props = ",".join(char.properties)
            print(f"  CHAR {char.uuid}  props=[{props}]  ({getattr(char, 'description', '')})")
            for desc in char.descriptors:
                print(f"    DESC {desc.uuid}")
    print("=" * 60 + "\n")


async def sniff_all_streams(device_address: str, duration=LISTEN_DURATION):
    print(f"Connecting to {device_address}...\n")

    async with BleakClient(device_address) as client:
        if not client.is_connected:
            print("Failed to connect.")
            return

        print("Connected!\nDiscovering services...\n")
        services = await resolve_services(client)

        if not services:
            print("Could not resolve services (empty).")
            print("Try: upgrade bleak, or re-run with Bluetooth toggled, or move device closer.")
            return

        await dump_gatt(services)

        notifiables = []
        for service in services:
            for char in service.characteristics:
                if "notify" in char.properties or "indicate" in char.properties:
                    notifiables.append(char.uuid)

        if not notifiables:
            print("No notify/indicate characteristics found.")
            return

        print("Subscribing to ALL notify/indicate characteristics:\n")
        for u in notifiables:
            print(" -", u)

        print("\nListening...\n" + "-" * 60)

        started = []
        for u in notifiables:
            try:
                await client.start_notify(u, make_notify_callback(u))
                started.append(u)
            except Exception as e:
                print(f"Could not subscribe to {u}: {e}")

        await asyncio.sleep(duration)

        for u in started:
            try:
                await client.stop_notify(u)
            except Exception:
                pass

        print("-" * 60)
        print("Finished listening.\n")


async def main():
    device = await find_polar_h10()
    if not device:
        return
    print(f"Auto-selecting: {device.name}\n")
    await sniff_all_streams(device.address, duration=LISTEN_DURATION)


if __name__ == "__main__":
    print("=" * 60)
    print("POLAR H10 STREAM SNIFFER (ALL NOTIFY/INDICATE OUTPUTS)")
    print("=" * 60)
    asyncio.run(main())