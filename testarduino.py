import serial
import time

PORT = "COM4"
BAUD = 115200

arduino = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # allow Arduino to reset after serial connection opens
print("Connected!")

def send_ttl_pulse():
    arduino.write(b"T")

def run_tonic_stimulation(n_trains=5, isi_s=5.0):

    for i in range(n_trains):
        send_ttl_pulse()
        time.sleep(isi_s)
    

def run_constant_stimulation(dur=45):

    try:
        for i in range(n_trains):
            send_ttl_pulse()
            time.sleep(isi_s)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("Done.")

run_tonic_stimulation(n_trains=5, isi_s=5.0)

arduino.close()
print("Closed.")