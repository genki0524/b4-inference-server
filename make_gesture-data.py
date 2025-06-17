from websockets.server import serve
import asyncio
import numpy as np
import cv2
import socket
HOST = socket.gethostname()
IP = socket.gethostbyname(HOST)
output_file = "Right_gesture.mp4"
frame_rate = 30
frame_size = (224,224)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file,fourcc,frame_rate,frame_size)

async def echo(websocket,path):
    async for message in websocket:
        arr = np.asarray(bytearray(message), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        img = cv2.resize(img,(224,224))
        out.write(img)
        cv2.imshow('image', img)
        cv2.waitKey(10)

async def main():
    try:
        async with serve(echo,IP,8008,max_size=100000000000000000):
            await asyncio.Future()  # run forever
    except KeyboardInterrupt:
        out.release()
        cv2.destroyAllWindows() 

asyncio.run(main())