import csv
from datetime import datetime
from gsv86lib import gsv86

COM_PORT = "COM16"
BAUD_RATE = 115200   # change to 230400 if supported

dev = gsv86(COM_PORT, BAUD_RATE)
dev.StartTransmission()

print("Recording at external 50 Hz trigger...")
print("Press Ctrl+C to stop.\n")

filename = f"gsv_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

last_timestamp = None

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["DeviceTime", "Ch0","Ch1","Ch2","Ch3","Ch4","Ch5"])

    try:
        while True:
            measurement = dev.ReadValue()

            if measurement and measurement.data:
                ts = measurement.getTimestamp()

                if ts != last_timestamp:
                    writer.writerow([
                        ts,
                        measurement.getChannel1(),
                        measurement.getChannel2(),
                        measurement.getChannel3(),
                        measurement.getChannel4(),
                        measurement.getChannel5(),
                        measurement.getChannel6()
                    ])

                    last_timestamp = ts

    except KeyboardInterrupt:
        print("\nStopped by user.")

dev.StopTransmission()
print("Saved:", filename)
