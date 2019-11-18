import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# データの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 小数化
x_train = x_train / 255
x_test = x_test / 255
# ViewData
fig = plt.figure(figsize = (8, 8))
fig.subplots_adjust(hspace=0, wspace=0)
for i in range(10):
    index, cnt = 0, 0
    while(True):
        if y_train[index] == i:
            ax = fig.add_subplot(10, 10, i*10+cnt+1, xticks=[], yticks=[])
            ax.imshow(x_train[index])
            cnt += 1
        if cnt >= 10: break
        index += 1
plt.show()