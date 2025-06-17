import asyncio
from websockets.server import serve
from websockets import connect
import numpy as np
import sys
import cv2
import json
import os
args = sys.argv

counter = 0

os.makedirs(f"./{args[1]}",exist_ok=True)
async def echo(websocket,path):
    global counter
    async for message in websocket:
        arr = np.asarray(bytearray(message), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        cv2.imshow('image', img)
        img = cv2.resize(img,(224,224))
        cv2.imwrite(f"./{args[1]}/{args[1]}_{counter}.jpg",img=img)
        counter += 1
        cv2.waitKey(10)
        
async def main():
    async with serve(echo,"172.16.1.12",8008,max_size=100000000000000000):
        await asyncio.Future()  # run forever

asyncio.run(main())