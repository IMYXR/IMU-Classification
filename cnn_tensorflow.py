import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 数据加载
AC = np.load('AC_array.npy')
AD = np.load('AD_array.npy')
BC = np.load('BC_array.npy')
BD = np.load('BD_array.npy')
# 为每个段分配标签
AC_labels = [0] * len(AC)
AD_labels = [1] * len(AD)
BC_labels = [2] * len(BC)
BD_labels = [3] * len(BD)
# 合并数据和标签
data =np.vstack([AC,AD,BC,BD])
labels = AC_labels + AD_labels+ BC_labels + BD_labels
X = np.array(data)
y = np.array(labels)


def bulid(X_train, y_train, batch_size=64, epochs=200):
    """
    搭建网络结构完成训练
    :param X_train: 训练集数据
    :param y_train: 训练集标签
    :param X_test: 测试集数据
    :param y_test: 测试集标签
    :param batch_size: 批次大小
    :param epochs: 循环轮数
    :return: acc和loss曲线
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=108, kernel_size=5, padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape=(18, 301)),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same',
                               activation=tf.keras.layers.ReLU()),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same',
                               activation=tf.keras.layers.ReLU()),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=108, activation=tf.keras.layers.ReLU()),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=4, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_train, y_train))
    model.summary()
    # 获得训练集和测试集的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 绘制acc曲线
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # 绘制loss曲线
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    bulid(X,y)

# 创建模型
# model = Sequential()
# model.add(Conv1D(filters=108, kernel_size=5, activation='relu', input_shape=(18, 41)))  # 修改输入形状为(18, 41)
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(108, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))  # 两个输出：正常和不正常
# # 编译模型
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # 训练模型
# model.fit(X, y, epochs=200, batch_size=64, validation_split=0.2)
# # 评估模型
# accuracy = model.evaluate(X, y)[1]
# print(f'Accuracy: {accuracy * 100:.2f}%')