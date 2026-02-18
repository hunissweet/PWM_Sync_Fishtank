import csv
from datetime import datetime
from gsv86lib import gsv86

# ----------------------------
# Configuration
# ----------------------------
COM_PORT_1 = "COM16"
COM_PORT_2 = "COM17"
BAUD_RATE = 115200  # increase if supported

# ----------------------------
# Open Devices
# ----------------------------
dev1 = gsv86(COM_PORT_1, BAUD_RATE)
dev2 = gsv86(COM_PORT_2, BAUD_RATE)

dev1.StartTransmission()
dev2.StartTransmission()

print("Dual sensor recording started (50 Hz trigger)")
print("Press Ctrl+C to stop.\n")

filename = f"dual_gsv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

last_ts1 = None

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)

    # Header (12 channels total)
    writer.writerow([
        "DeviceTime",
        "S1_Ch1","S1_Ch2","S1_Ch3","S1_Ch4","S1_Ch5","S1_Ch6",
        "S2_Ch1","S2_Ch2","S2_Ch3","S2_Ch4","S2_Ch5","S2_Ch6"
    ])

    try:
        while True:

            # ---- Read sensor 1 ----
            m1 = dev1.ReadValue()

            if m1 and m1.data:
                ts1 = m1.getTimestamp()

                # Ignore duplicates
                if ts1 != last_ts1:

                    # ---- Read sensor 2 ----
                    m2 = dev2.ReadValue()

                    if m2 and m2.data:

                        row = [
                            ts1,
                            m1.getChannel1(),
                            m1.getChannel2(),
                            m1.getChannel3(),
                            m1.getChannel4(),
                            m1.getChannel5(),
                            m1.getChannel6(),
                            m2.getChannel1(),
                            m2.getChannel2(),
                            m2.getChannel3(),
                            m2.getChannel4(),
                            m2.getChannel5(),
                            m2.getChannel6(),
                        ]

                        writer.writerow(row)
                        f.flush()

                        last_ts1 = ts1

    except KeyboardInterrupt:
        print("\nStopped by user.")

dev1.StopTransmission()
dev2.StopTransmission()

print("Saved:", filename)
