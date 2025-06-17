import numpy as np
import cv2

net = cv2.dnn.readNetFromONNX("model.onnx")

img = cv2.imread("GUU_Light_046.jpg")

img = cv2.resize(img,(224,224))

img = img.astype(np.float32)/255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

img = (img-mean)/std

img = img[:,:,::-1]
img = img.transpose(2,0,1)
img = np.expand_dims(img,0)
print(img.shape)

net.setInput(img)

output = net.forward()

print(output)