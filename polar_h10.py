#!/usr/bin/env python3
"""
Polar H10 Heart Rate Monitor Script
Connects to Polar H10 and streams real-time heart rate data
"""

import asyncio
from bleak import BleakScanner, BleakClient

# Your Polar H10 Device ID
POLAR_H10_ID = "0C83F23F"

# Standard Bluetooth Heart Rate Service UUID
HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


def heart_rate_callback(sender, data):
    """
    Callback function to handle heart rate data
    Heart rate data format (as per Bluetooth spec):
    - Byte 0: Flags
    - Byte 1: Heart Rate Value (if uint8)
    - Bytes 1-2: Heart Rate Value (if uint16)
    """
    # First byte contains flags
    flags = data[0]
    
    # Check if heart rate is in uint8 or uint16 format
    if flags & 0x01:
        # Heart rate is uint16 (2 bytes)
        hr_value = int.from_bytes(data[1:3], byteorder='little')
    else:
        # Heart rate is uint8 (1 byte)
        hr_value = data[1]
    
    print(f"❤️  Heart Rate: {hr_value} BPM")


async def find_polar_h10(timeout=10):
    """Scan for Polar H10 devices"""
    print(f"Scanning for Polar H10 (ID: {POLAR_H10_ID}) for {timeout} seconds...")
    print("Make sure your Polar H10 is turned on and not connected to other devices!\n")
    
    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
    
    # First, try to find your specific H10 by ID
    for address, (device, adv_data) in devices.items():
        if device.name and POLAR_H10_ID in device.name:
            rssi = adv_data.rssi if hasattr(adv_data, 'rssi') else 'N/A'
            print(f"✓ Found your Polar H10: {device.name}")
            print(f"  Address: {device.address}")
            print(f"  RSSI: {rssi} dBm\n")
            return [device]
    
    # If not found by ID, filter for any Polar devices
    polar_devices = []
    for address, (device, adv_data) in devices.items():
        if device.name and "polar" in device.name.lower():
            polar_devices.append((device, adv_data))
    
    if not polar_devices:
        print("No Polar devices found.")
        print("\nTroubleshooting:")
        print("1. Make sure the H10 is turned on (LED should blink)")
        print("2. Disconnect from any other apps (Polar Beat, etc.)")
        print("3. Try moving the sensor closer to your computer")
        print("4. Make sure the strap is properly worn (moistened electrodes)")
        return None
    
    print(f"Found {len(polar_devices)} Polar device(s) (but not your specific H10 ID):\n")
    for i, (device, adv_data) in enumerate(polar_devices):
        rssi = adv_data.rssi if hasattr(adv_data, 'rssi') else 'N/A'
        print(f"{i + 1}. {device.name}")
        print(f"   Address: {device.address}")
        print(f"   RSSI: {rssi} dBm\n")
    
    return [d for d, _ in polar_devices]


async def connect_and_stream_hr(device_address, duration=30):
    """Connect to Polar H10 and stream heart rate data"""
    print(f"Connecting to {device_address}...")
    
    try:
        async with BleakClient(device_address) as client:
            is_connected = client.is_connected
            print(f"Connected: {is_connected}\n")
            
            if not is_connected:
                print("Failed to connect.")
                return
            
            # Check if heart rate service is available
            services = client.services
            hr_service = None
            for service in services:
                if service.uuid.lower() == HEART_RATE_SERVICE_UUID.lower():
                    hr_service = service
                    break
            
            if not hr_service:
                print("Heart Rate service not found on this device.")
                return
            
            print("Heart Rate service found!")
            print(f"Starting heart rate monitoring for {duration} seconds...")
            print("Put on the H10 strap and start moving to see your heart rate!\n")
            print("-" * 40)
            
            # Subscribe to heart rate notifications
            await client.start_notify(HEART_RATE_MEASUREMENT_UUID, heart_rate_callback)
            
            # Stream for specified duration
            await asyncio.sleep(duration)
            
            # Stop notifications
            await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
            print("-" * 40)
            print("\nStopped monitoring.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. The H10 is properly worn (moistened electrodes)")
        print("2. No other app is connected to the H10")
        print("3. You have Bluetooth permissions on your system")

async def dump_gatt(device_address):
    async with BleakClient(device_address) as client:
        await client.connect()
        print("Connected:", client.is_connected)

        # Ensure services are resolved
        services = await client.get_services()

        for service in services:
            print(f"\nSERVICE {service.uuid} ({service.description})")
            for char in service.characteristics:
                props = ",".join(char.properties)
                print(f"  CHAR {char.uuid}  props=[{props}]  ({char.description})")

                # Also show descriptors (sometimes useful)
                for desc in char.descriptors:
                    print(f"    DESC {desc.uuid}")


async def main():
    """Main function"""
    # Scan for Polar H10
    devices = await find_polar_h10(timeout=10)
    
    if not devices:
        return
    
    # Let user choose a device or auto-select if only one
    selected_device = None
    
    if len(devices) == 1:
        selected_device = devices[0]
        print(f"Auto-selecting: {selected_device.name}")
    else:
        try:
            choice = input("Enter device number to connect (or 'q' to quit): ")
            if choice.lower() == 'q':
                return
            
            device_index = int(choice) - 1
            if 0 <= device_index < len(devices):
                selected_device = devices[device_index]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return
    
    if selected_device:
        print()
        # Connect and stream for 30 seconds (you can change this)
        await connect_and_stream_hr(selected_device.address, duration=30)
        await dump_gatt(selected_device.address)


if __name__ == "__main__":
    print("=" * 50)
    print("Polar H10 Heart Rate Monitor")
    print("=" * 50)
    print()
    asyncio.run(main())