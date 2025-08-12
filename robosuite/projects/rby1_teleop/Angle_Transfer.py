import math
from pySerialTransfer import pySerialTransfer as txfer
import time

class Angle_Transfer:
    def __init__(self, serial_port, baud=9600):
        self.link = txfer.SerialTransfer(serial_port, baud)
        if not self.link.open():
            raise IOError(f"Failed to open serial port: {serial_port}")
        time.sleep(1)  # Wait for the serial connection to initialize

    def is_connected(self):
        return self.link.connection.is_open

    def send_face(self, angle1, angle2):
        if self.is_connected():
            data = [angle1, angle2]
            size = self.link.tx_obj(data)
            self.link.send(size)
        else:
            print("Serial connection is not open.")

    def close(self):
        if self.is_connected():
            self.link.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    # serial port used to transmit data to Arduino
    serial_port = '/dev/cu.usbmodem1101'
    transfer = None
    try:
        transfer = Angle_Transfer(serial_port, baud=9600)
        pos1 = 90
        pos2 = 90

        end_time = time.time() + 20 # Run for 20 seconds
        while time.time() < end_time:
            transfer.send_face(pos1, pos2)
            pos1 = 90 + int(90 * math.sin(time.time()))
            pos2 = 90 + int(30 * math.cos(time.time()))
            time.sleep(0.1)  # Adjust the sleep time as needed
            print(f"Sent angles: {pos1}, {pos2}")

    except (IOError, KeyboardInterrupt) as e:
        print(f"An error occurred: {e}")
    finally:
        if transfer:
            transfer.close()