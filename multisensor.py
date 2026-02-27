#!/usr/bin/env python3
"""
Triple-Sensor LSL Streamer (Polar HR + Vernier Respiration + Tobii Eye Tracker)
===============================================================================

Synchronized physiological and oculomotor monitoring with LSL streaming.
Heart Rate + Respiration + Eye Tracking data collection.
"""

import asyncio
import time
import threading
from pylsl import StreamInfo, StreamOutlet, resolve_streams, StreamInlet
from bleak import BleakScanner, BleakClient
from godirect import GoDirect
import numpy as np
from datetime import datetime

# Polar H10 Configuration
POLAR_H10_ID = "0C83F23F"
HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

class TripleSensorLSLStreamer:
    def __init__(self):
        # Polar setup
        self.polar_client = None
        self.polar_outlet = None
        self.polar_streaming = False
        self.polar_device = None
        
        # Vernier setup
        self.vernier_device = None
        self.vernier_outlet = None
        self.vernier_streaming = False
        self.vernier_thread = None
        
        # Tobii setup
        self.tobii_eyetracker = None
        self.tobii_outlet = None
        self.tobii_streaming = False
        
        # Recording
        self.recording = False
        self.recorded_data = {}
        self.start_time = None
    # === POLAR H10 METHODS ===
    async def find_polar_h10(self, timeout=15):
        """Find Polar H10 using extended timeout"""
        print(f"🔍 Scanning for Polar H10 (ID: {POLAR_H10_ID}) for {timeout} seconds...")
        
        try:
            devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
            
            # First, try to find your specific H10 by ID
            for address, (device, adv_data) in devices.items():
                if device.name and POLAR_H10_ID in device.name:
                    print(f"Found Polar H10: {device.name}")
                    return device
            
            # If not found by ID, look for any Polar device
            for address, (device, adv_data) in devices.items():
                if device.name and "polar" in device.name.lower():
                    print(f"Found Polar device: {device.name}")
                    return device
            
            print(" No Polar devices found")
            return None
            
        except Exception as e:
            print(f"Polar scan error: {e}")
            return None

    def heart_rate_callback(self, sender, data):
        """Handle heart rate data from Polar H10 (HR + RR intervals when present)"""
        if not self.polar_streaming or not self.polar_outlet:
            return
            
        try:
            flags = data[0]
            idx = 1

            # HR value (uint8 vs uint16)
            if flags & 0x01:
                hr_value = int.from_bytes(data[idx:idx+2], byteorder='little')
                idx += 2
            else:
                hr_value = data[idx]
                idx += 1

            # RR intervals present?
            rr_ms = []
            if flags & 0x10:
                # Remaining bytes are RR-intervals (uint16 little-endian), units = 1/1024 sec
                while idx + 1 < len(data):
                    rr_1024 = int.from_bytes(data[idx:idx+2], byteorder='little')
                    idx += 2
                    rr_ms.append(rr_1024 * 1000.0 / 1024.0)

            # Push to LSL:
            # - Keep your existing HR stream name/type but make it 2 channels: [HR_BPM, RR_MS]
            # - If multiple RR intervals arrive in one packet, push multiple samples (same HR, different RR)
            if rr_ms:
                for rr in rr_ms:
                    self.polar_outlet.push_sample([float(hr_value), float(rr)])
            else:
                # No RR in this packet -> use -1 as "missing" marker
                self.polar_outlet.push_sample([float(hr_value), -1.0])
            
            if hasattr(self, '_hr_count'):
                self._hr_count += 1
            else:
                self._hr_count = 1
                
        except Exception as e:
            print(f"Heart rate parsing error: {e}")

    async def start_polar_stream(self):
        """Start Polar H10 heart rate stream (now includes RR intervals)"""
        try:
            self.polar_device = await self.find_polar_h10(timeout=15)
            if not self.polar_device:
                print(" Polar device not found")
                return False
            
            print(f"Connecting to Polar at {self.polar_device.address}...")
            
            self.polar_client = BleakClient(self.polar_device.address)
            await self.polar_client.connect()
            
            if not self.polar_client.is_connected:
                print(" Failed to connect to Polar")
                return False
            
            # Check for heart rate service
            hr_service = None
            for service in self.polar_client.services:
                if service.uuid.lower() == HEART_RATE_SERVICE_UUID.lower():
                    hr_service = service
                    break
            
            if not hr_service:
                print("❌ Heart Rate service not found")
                return False
            
            # Create LSL stream for heart rate + RR
            # Channels: HR_BPM, RR_MS
            info = StreamInfo('PolarHeartRate', 'HeartRate', 2, 1, 'float32', 'polar_hr')
            info.desc().append_child_value("manufacturer", "Polar")
            info.desc().append_child_value("model", "H10")
            info.desc().append_child_value("channels", "hr_bpm,rr_ms")
            info.desc().append_child_value("units", "BPM,ms")
            self.polar_outlet = StreamOutlet(info)
            
            # Subscribe to heart rate notifications
            await self.polar_client.start_notify(HEART_RATE_MEASUREMENT_UUID, self.heart_rate_callback)
            
            self.polar_streaming = True
            print("✅ Polar heart rate + RR stream started")
            return True
            
        except Exception as e:
            print(f"❌ Polar connection error: {e}")
            return False
    
    # === VERNIER METHODS ===
    def start_vernier_stream(self):
        """Start Vernier respiration stream"""
        try:
            print("🔧 Connecting to Vernier USB device...")
            go_direct = GoDirect(use_ble=False, use_usb=True)
            devices = go_direct.list_devices()
            
            if devices:
                self.vernier_device = devices[0]
                if self.vernier_device.open():
                    print(f"✅ Connected to: {self.vernier_device.name}")
                    
                    self.vernier_device.enable_sensors([1])
                    
                    info = StreamInfo('VernieRespiration', 'Respiration', 1, 10, 'float32', 'vernier_resp')
                    info.desc().append_child_value("manufacturer", "Vernier")
                    info.desc().append_child_value("model", "Go Direct Respiration Belt")
                    info.desc().append_child_value("units", "Newtons")
                    self.vernier_outlet = StreamOutlet(info)
                    
                    self.vernier_streaming = True
                    self.vernier_thread = threading.Thread(target=self._vernier_stream_worker)
                    self.vernier_thread.start()
                    
                    print("✅ Vernier respiration stream started (10 Hz)")
                    return True
            
            print("No Vernier USB devices found")
            return False
            
        except Exception as e:
            print(f" Vernier error: {e}")
            return False
    
    def _vernier_stream_worker(self):
        """Vernier streaming worker thread"""
        if not self.vernier_device:
            return
            
        self.vernier_device.start()
        sample_count = 0
        
        while self.vernier_streaming:
            try:
                if self.vernier_device.read():
                    sensors = self.vernier_device.get_enabled_sensors()
                    for sensor in sensors:
                        if hasattr(sensor, 'value') and sensor.value is not None:
                            self.vernier_outlet.push_sample([sensor.value])
                            sample_count += 1
                            
                            if sample_count % 100 == 0:
                                print(f"🫁 Respiration samples: {sample_count}")
                
                time.sleep(0.1)  # 10 Hz
                
            except Exception as e:
                print(f"Vernier streaming error: {e}")
                break
    
    # === TOBII EYE TRACKER METHODS ===
    def start_tobii_stream(self):
        """Start Tobii eye tracking stream"""
        try:
            import tobii_research as tr
            
            print("🔧 Connecting to Tobii eye tracker...")
            
            # Find eye tracker
            found_eyetrackers = tr.find_all_eyetrackers()
            if not found_eyetrackers:
                print("No Tobii eye tracker found")
                return False
            
            self.tobii_eyetracker = found_eyetrackers[0]
            print(f"✅ Found Tobii: {self.tobii_eyetracker.device_name}")
            print(f"   Model: {self.tobii_eyetracker.model}")
            print(f"   Serial: {self.tobii_eyetracker.serial_number}")
            
            # Get sampling frequency
            sampling_freq = self.tobii_eyetracker.get_gaze_output_frequency()
            print(f"   Sampling rate: {sampling_freq} Hz")
            
            # Create LSL stream for eye tracking
            # Channels: gaze_x, gaze_y, pupil_left, pupil_right, validity_left, validity_right
            info = StreamInfo('TobiiEyeTracker', 'Gaze', 6, sampling_freq, 'float32', 'tobii_gaze')
            info.desc().append_child_value("manufacturer", "Tobii")
            info.desc().append_child_value("model", self.tobii_eyetracker.model)
            info.desc().append_child_value("channels", "gaze_x,gaze_y,pupil_left,pupil_right,valid_left,valid_right")
            info.desc().append_child_value("units", "normalized_coords,mm,binary")
            self.tobii_outlet = StreamOutlet(info)
            
            # Subscribe to gaze data
            self.tobii_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
            
            self.tobii_streaming = True
            print(f"Tobii eye tracking stream started ({sampling_freq} Hz)")
            return True
            
        except ImportError:
            print(" Tobii Research SDK not installed")
            return False
        except Exception as e:
            print(f" Tobii connection error: {e}")
            return False
    
    def gaze_data_callback(self, gaze_data):
        """Extract REAL Tobii data - correct structure"""
        if not self.tobii_streaming or not self.tobii_outlet:
            return
        
        try:
            # Extract gaze coordinates from display area (0-1 coordinates)
            left_gaze = gaze_data.left_eye.gaze_point.position_on_display_area
            right_gaze = gaze_data.right_eye.gaze_point.position_on_display_area
            
            # Extract validity
            left_valid = gaze_data.left_eye.gaze_point.validity
            right_valid = gaze_data.right_eye.gaze_point.validity
            
            # Calculate average gaze position
            if left_valid and right_valid:
                gaze_x = (left_gaze[0] + right_gaze[0]) / 2
                gaze_y = (left_gaze[1] + right_gaze[1]) / 2
            elif left_valid:
                gaze_x, gaze_y = left_gaze[0], left_gaze[1]
            elif right_valid:
                gaze_x, gaze_y = right_gaze[0], right_gaze[1]
            else:
                gaze_x, gaze_y = -1.0, -1.0  # Invalid data marker
            
            # Extract pupil data (need to explore what's inside pupil object)
            try:
                pupil_left = gaze_data.left_eye.pupil.diameter
                pupil_right = gaze_data.right_eye.pupil.diameter
            except:
                pupil_left = pupil_right = -1.0  # If pupil data structure is different
            
            # Validity flags
            validity_left = 1.0 if left_valid else 0.0
            validity_right = 1.0 if right_valid else 0.0
            
            # Send real data to LSL
            sample = [gaze_x, gaze_y, pupil_left, pupil_right, validity_left, validity_right]
            self.tobii_outlet.push_sample(sample)
            
        except Exception as e:
            print(f"Gaze extraction error: {e}")
    
    # === MAIN CONTROL METHODS ===
    async def start_all_streams(self):
        """Start all three sensor streams"""
        
        # Start Vernier (10 Hz)
        print("\n1️⃣ Starting Vernier respiration monitoring...")
        vernier_ok = self.start_vernier_stream()
        
        # Start Polar (1 Hz)
        print("\n2️⃣ Starting Polar heart rate monitoring...")
        polar_ok = await self.start_polar_stream()
        
        # Start Tobii (60+ Hz)
        print("\n3️⃣ Starting Tobii eye tracking...")
        tobii_ok = self.start_tobii_stream()
        
        print("\n" + "=" * 70)
        success_count = sum([vernier_ok, polar_ok, tobii_ok])
        
        if success_count == 3:
            print("ALL THREE SENSORS CONNECTED SUCCESSFULLY!")
            return True
        elif success_count >= 1:
            print(f" PARTIAL SUCCESS: {success_count}/3 sensors connected")
            if vernier_ok:
                print("✅ Respiration: Available")
            if polar_ok:
                print("✅ Heart Rate: Available")
            if tobii_ok:
                print("✅ Eye Tracking: Available")
            return True
        else:
            print(" ALL SENSORS FAILED TO CONNECT")
            return False
    
    def start_recording(self, output_prefix="triple_sensor"):
        print("\nSearching for LSL streams...")
        time.sleep(3)
        
        streams = resolve_streams()
        if not streams:
            print("No LSL streams found!")
            return False

        print(f"Found {len(streams)} LSL streams:")
        self.inlets = []
        for stream in streams:
            print(f"  {stream.name()} ({stream.type()}) @ {stream.nominal_srate()} Hz")
            inlet = StreamInlet(stream, max_buflen=360)
            self.inlets.append({'inlet': inlet, 'name': stream.name(),
                                'type': stream.type(), 'data': [], 'timestamps': []})

        self.recording = True
        self.start_time = time.time()
        
        for inlet_data in self.inlets:
            thread = threading.Thread(target=self._record_stream, args=(inlet_data,))
            thread.daemon = True
            thread.start()

        print("\n🔴 RECORDING — press Ctrl+C to stop and save\n")
        return True
        
        # Auto-stop with progress updates
        def auto_stop():
            for i in range(duration_seconds):
                if not self.recording:
                    break
                time.sleep(1)
                remaining = duration_seconds - i - 1
                if remaining % 10 == 0 or remaining <= 5:
                    print(f" Recording... {remaining}s remaining")
            
            if self.recording:
                self.stop_recording(output_prefix)
        
        stop_thread = threading.Thread(target=auto_stop)
        stop_thread.daemon = True
        stop_thread.start()
        
        return True
    
    def _record_stream(self, inlet_data):
        """Record from one stream"""
        inlet = inlet_data['inlet']
        
        while self.recording:
            try:
                sample, timestamp = inlet.pull_sample(timeout=0.1)
                if sample is not None:
                    inlet_data['data'].append(sample)
                    inlet_data['timestamps'].append(timestamp)
            except:
                break
    
    def stop_recording(self, output_prefix="triple_sensor"):
        """Stop recording and save data"""
        if not self.recording:
            return
        
        print("\n⏹️  STOPPING RECORDING...")
        self.recording = False
        
        for thread in self.record_threads:
            thread.join(timeout=2)
        
        filename = self.save_data(output_prefix)
        
        duration = time.time() - self.start_time
        print("=" * 70)
        print("RECORDING COMPLETED")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Data saved: {filename}")
        print("=" * 70)
        
        return filename
    
    def save_data(self, output_prefix="triple_sensor"):
        """Save recorded data to NPZ format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_prefix}_{timestamp}.npz"
        
        data_dict = {}
        total_samples = 0
        
        for inlet_data in self.inlets:
            if inlet_data['data']:
                data_array = np.array(inlet_data['data'])
                timestamps_array = np.array(inlet_data['timestamps'])
                
                data_dict[inlet_data['name']] = {
                    'data': data_array,
                    'timestamps': timestamps_array,
                    'type': inlet_data['type']
                }
                
                duration = timestamps_array[-1] - timestamps_array[0] if len(timestamps_array) > 1 else 0
                rate = len(inlet_data['data']) / duration if duration > 0 else 0
                
                print(f"{inlet_data['name']}: {len(inlet_data['data'])} samples ({rate:.1f} Hz)")
                total_samples += len(inlet_data['data'])
        
        if data_dict:
            np.savez(filename, **data_dict)
            print(f"Total samples: {total_samples}")
        
        return filename
    
    async def stop_all_streams(self):
        """Stop all streams and cleanup"""
        print("\nCleaning up streams...")
        
        if self.recording:
            self.stop_recording()
        
        # Stop Vernier
        if self.vernier_streaming:
            self.vernier_streaming = False
            if self.vernier_thread and self.vernier_thread.is_alive():
                self.vernier_thread.join(timeout=3)
            if self.vernier_device:
                self.vernier_device.stop()
                self.vernier_device.close()
            print("Vernier stream stopped")
        
        # Stop Polar
        if self.polar_streaming:
            self.polar_streaming = False
            if self.polar_client and self.polar_client.is_connected:
                await self.polar_client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
                await self.polar_client.disconnect()
            print("Polar stream stopped")
        
        # Stop Tobii
        if self.tobii_streaming:
            self.tobii_streaming = False
            if self.tobii_eyetracker:
                try:
                    import tobii_research as tr
                    self.tobii_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
                except:
                    pass
            print("Tobii stream stopped")

# Main function
async def main():
    """Main triple-sensor monitoring session"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Triple-Sensor Physiological Monitor')
    parser.add_argument('--duration', '-d', type=int, default=300, 
                        help='Recording duration in seconds (default: 60)')
    parser.add_argument('--output', '-o', type=str, default='triple_sensor_session',
                        help='Output file prefix (default: triple_sensor_session)')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Test connections only (no recording)')
    
    args = parser.parse_args()
    
    streamer = TripleSensorLSLStreamer()
    
    try:
        # Start all streams
        if not await streamer.start_all_streams():
            print("\n FAILED TO START STREAMS")
            return 1
        
        if args.test:
            print("\nCONNECTION TEST COMPLETED")
            print("All working sensors are streaming. Use --duration to record data.")
            await asyncio.sleep(10)
        else:
            # Wait for streams to stabilize
            print(f"\nStabilizing streams for 3 seconds...")
            await asyncio.sleep(3)
            
            # Start recording
            print(f"\n Starting {args.duration}-second recording...")
            if streamer.start_recording(args.output):
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nCtrl+C received — stopping...")
                    streamer.stop_recording(args.output)
        
        print("\n🎉 TRIPLE-SENSOR SESSION COMPLETED SUCCESSFULLY!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n SESSION INTERRUPTED BY USER")
        return 0
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        return 1
        
    finally:
        # Always cleanup
        await streamer.stop_all_streams()

if __name__ == "__main__":
    import sys
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)