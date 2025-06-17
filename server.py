import asyncio
from websockets.server import serve
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import json
import socket
HOST = socket.gethostname()
IP = socket.gethostbyname(HOST)
from scipy.stats import entropy
classes = ["UP","DOWN","LEFT","RIGHT","FORWARD"]
net = cv2.dnn.readNetFromONNX("regularizedModel_2024-09-27.onnx")
result_buffer = []

def get_local_ip():
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8",80))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = "127.0.0.1"
    return local_ip

def inference(net,img):
    net.setInput(img)
    output = net.forward()
    output = torch.from_numpy(output.astype(np.float32)).clone()
    output = F.softmax(output, dim=1)
    output = output.to('cpu').detach().numpy().copy()
    entropy_value = entropy(np.squeeze(output))
    return entropy_value,output

async def echo(websocket,path):
    async for message in websocket:
        try:
            if isinstance(message,bytes):
                    arr = np.asarray(bytearray(message), dtype=np.uint8)
                    img = cv2.imdecode(arr, -1)
                    global result_buffer
                    if img is not None and img.size > 0:
                        img = cv2.resize(img,(224,224))
                        img = img.astype(np.float32)/255.0
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img = (img-mean)/std
                        img = img[:,:,::-1]
                        img = img.transpose(2,0,1)
                        img = np.expand_dims(img,0)
                        entropy_value,output = inference(net=net,img=img)
                        if entropy_value > 0.5:
                            result = "Nothing"
                        else:
                            result = classes[np.argmax(output)]
                        result_buffer.append(result)
                        if len(result_buffer) >= 5:
                            temp_result_buffer = result_buffer.copy()
                            result_buffer.clear()
                            if all(val == temp_result_buffer[0] for val in temp_result_buffer):
                                await websocket.send(json.dumps({"pose":result}))
                        print("--------------------")
                        print("result: ",result)
                        print(f"UP: ",output[0][0],"\nDOWN: ",output[0][1], "\nLEFT: ",output[0][2], " \nRIGHT: ",output[0][3], "\nFORWARD: ", output[0][4])
                        print("--------------------")
                    else:
                        print("映像を受け取れませんでした。")
            else:
                raise TypeError
        except TypeError as e:
            print(message)
            await websocket.send(message)

async def main():
    async with serve(echo,get_local_ip(),8008,max_size=100000000000000000):
        await asyncio.Future()  # run forever

asyncio.run(main())