import asyncio
import json
import socket

from websockets.server import serve
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import json
from scipy.stats import entropy

#モデルの読み込み
model_path = ""
net = cv2.dnn.readNetFromONNX(model_path)

#推論結果のリスト
classes = ["UP","DOWN","LEFT","RIGHT","FORWARD"]

#推論結果を格納する変数
result_buffer = []

def get_local_ip() -> str:
    """
    :return: ローカルIPアドレスを返す
    """
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8",80))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = "127.0.0.1"
    return local_ip

def inference(net,img) -> tuple[np.ndarray,np.ndarray]:
    """画像を推論してその結果を出力

    :param cv2.dnn_Net net: 推論モデル
    :param numpy.ndaary img: 推論される画像データ  
    :return: entoropyが予測の信頼度、outputが各クラスの推論結果の確率
    """
    net.setInput(img)
    output = net.forward()

    #outputをTensorに変換する
    output = torch.from_numpy(output.astype(np.float32)).clone()

    #softmax関数を通す
    output = F.softmax(output, dim=1)

    #outputをcpuに載せてndarryに変換する
    output = output.to('cpu').detach().numpy().copy()

    #outputの信頼度を取得
    entropy_value = entropy(np.squeeze(output))
    return entropy_value,output

def preprocess_image(input_img:np.ndarray) -> np.ndarray:
    """取得したデータをinference関数で利用できるように前処理

    :param input_img np.ndarry: 入力画像
    :return: inference関数が利用できるように前処理したndarryのデータ
    """
    img = input_img.copy()

    #モデルが読み込めるように加工
    img = cv2.resize(img,(224,224))
    img = img.astype(np.float32)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img-mean)/std

    # BGR→RGB
    img = img[:,:,::-1]

    # (H, W, C)→(C, H, W)
    img = img.transpose(2,0,1)

    # バッチ次元追加
    img = np.expand_dims(img,0)
    return img

async def handle_inference(websocket):
    async for message in websocket:
        
        #受信したデータがbyteデータの場合
        if isinstance(message,bytes):  

            #受信したデータをndarryに復元する
            arr = np.asarray(bytearray(message), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)

            #resultの結果を格納する変数
            global result_buffer

            if img is not None and img.size > 0:
                img = preprocess_image(img)
                entropy_value,output = inference(net=net,img=img)
                if entropy_value > 0.5:
                    result = "Nothing"
                else:
                    result = classes[np.argmax(output)]
                result_buffer.append(result)
                if len(result_buffer) >= 5:
                    temp_result_buffer = result_buffer.copy()
                    result_buffer.clear()

                    #bufferに格納されているデータが全て同じ結果だった時poseが確定する
                    if all(val == temp_result_buffer[0] for val in temp_result_buffer):
                        await websocket.send(json.dumps({"pose":result}))
                
                print("--------------------")
                print("result: ",result)
                print(f"UP: ",output[0][0],"\nDOWN: ",output[0][1], "\nLEFT: ",output[0][2], " \nRIGHT: ",output[0][3], "\nFORWARD: ", output[0][4])
                print("--------------------")
            else:
                print("映像を受け取れませんでした。")

        #受信したデータがジェスチャセンサのデータの場合
        else:
            print(message)
            await websocket.send(message)

async def main():
    async with serve(handle_inference,get_local_ip(),8008,max_size=100000000000000000):
        await asyncio.Future()  # run forever

asyncio.run(main())