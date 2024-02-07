import time
from robomaster import conn
from MyQR import myqr
from PIL import Image

QRCODE_NAME = "qrcode.png"

if __name__ == '__main__':

    # Manually indicate the IP of your computer if it is not found automatically
    # robomaster.config.LOCAL_IP_STR = "XXX.XXX.XXX.XXX"

    # Create QR to connect the robot to your Wi-Fi
    helper = conn.ConnectionHelper()
    # info = helper.build_qrcode_string(ssid="ssid_of_your_router", password="wifi_password")
    info = helper.build_qrcode_string(ssid="GTI", password="tres delicias")
    myqr.run(words=info)

    # Show the QR
    img = Image.open(QRCODE_NAME)
    img.show()

    # Wait for connection (scan the qr with the robot)
    print("Waiting for connections...")
    time.sleep(5)
    if helper.wait_for_connection():
        print("Connected!")
    else:
        print("Connect failed!")
