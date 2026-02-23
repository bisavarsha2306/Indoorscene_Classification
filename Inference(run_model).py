import numpy as np
from tensorflow.keras.models import Model, load_model
import cv2

model = load_model(r'D:\mohanteja\python_files\model.h5')
img = cv2.imread(r'D:\mohanteja\python_files\dataset_1\train\airport_inside\airport_inside_0003.jpg')

img=cv2.resize(img,(224,224))
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
# plt.imshow(img)
# plt.show()
pred = model.predict(img[np.newaxis, :, :, :])
pred_class = np.argmax(pred)
print(pred_class)

