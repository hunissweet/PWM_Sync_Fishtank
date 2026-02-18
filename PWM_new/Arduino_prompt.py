import argparse

import time

import serial





def open_serial(port: str, baud: int = 115200, timeout: float = 0.5) -> serial.Serial:

    ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)

    # Mega 2560（USB 串口）经常会在打开串口时复位，等它启动并打印欢迎信息

    time.sleep(2.0)

    ser.reset_input_buffer()

    return ser





def send_cmd(ser: serial.Serial, cmd: str) -> str:

    ser.write((cmd.strip() + "\n").encode("ascii", errors="ignore"))

    ser.flush()

    # 读回一行回应（如果你代码会回显 OK: ...）

    try:

        line = ser.readline().decode("utf-8", errors="replace").strip()

    except Exception:

        line = ""

    return line





def main():

    ap = argparse.ArgumentParser(description="Set RF2040 trigger frequency via serial.")

    ap.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0 (Linux) or COM5 (Windows)")

    ap.add_argument("--baud", type=int, default=115200)

    ap.add_argument("--freq", type=float, required=True, help="Frequency in Hz, e.g. 100")

    ap.add_argument("--pulse_us", type=int, default=None, help="Pulse width in us, e.g. 100")

    args = ap.parse_args()



    ser = open_serial(args.port, args.baud)



    if args.pulse_us is None:

        cmd = f"{args.freq:.6g}"          # 只发频率：例如 "100"

    else:

        cmd = f"F {args.freq:.6g} P {args.pulse_us}"  # 例如 "F 100 P 100"



    resp = send_cmd(ser, cmd)

    print("Sent:", cmd)

    if resp:

        print("Recv:", resp)

    else:

        print("Recv: (no response)")



    ser.close()





if __name__ == "__main__":

    main()
 