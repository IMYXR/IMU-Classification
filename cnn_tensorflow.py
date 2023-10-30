import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# 数据加载
AC = np.loadtxt(r'AC.csv',delimiter=',')
AD = np.loadtxt(r'AD.csv',delimiter=',')
BC = np.loadtxt(r'BC.csv',delimiter=',')
BD = np.loadtxt(r'BD.csv',delimiter=',')
AC = AC.T
AD = AD.T
BC = BC.T
BD = BD.T
def detect_local_maxima(data, prominence_value,height_value):
    # 使用find_peaks找到局部最大值的位置
    local_maxima, _ = find_peaks(data[8, :], prominence=prominence_value,height=height_value)
    return local_maxima

# 对每个数据集单独设置find_peaks的参数
AC_prominence = 8
AD_prominence = 30
BC_prominence = 30
BD_prominence = 25
AC_height = 320
AD_height = 340
BC_height = 380
BD_height = 0
AC_maxima = detect_local_maxima(AC, AC_prominence,AC_height)
AD_maxima = detect_local_maxima(AD, AD_prominence,AD_height)
BC_maxima = detect_local_maxima(BC, BC_prominence,BC_height)
BD_maxima = detect_local_maxima(BD, BD_prominence,BD_height)

def extract_segments_around_maxima(data, maxima_indices):
    segments = []
    for max_idx in maxima_indices:
        start_idx = max(0, max_idx - 100)
        end_idx = min(data.shape[1], max_idx + 101)  # +101以确保形状为(18, 201)
        segment = data[:, start_idx:end_idx]

        # 如果段的大小小于201，我们需要填充它
        if segment.shape[1] < 201:
            padding = np.zeros((18, 201 - segment.shape[1]))
            segment = np.hstack([segment, padding])

        segments.append(segment)
    return segments


# 使用上述函数为每个数据集提取段
AC_segments = extract_segments_around_maxima(AC, AC_maxima)
AD_segments = extract_segments_around_maxima(AD, AD_maxima)
BC_segments = extract_segments_around_maxima(BC, BC_maxima)
BD_segments = extract_segments_around_maxima(BD, BD_maxima)

# 如果需要，您可以将这些段转换为numpy数组
AC_segments_array = np.array(AC_segments)
AD_segments_array = np.array(AD_segments)
BC_segments_array = np.array(BC_segments)
BD_segments_array = np.array(BD_segments)

# 为每个段分配标签
AC_labels = [0] * len(AC_segments_array)
AD_labels = [1] * len(AD_segments_array)
BC_labels = [2] * len(BC_segments_array)
BD_labels = [3] * len(BD_segments_array)
# 合并数据和标签
data = AC_segments + AD_segments+ BC_segments+ BD_segments
labels = AC_labels + AD_labels+ BC_labels + BD_labels
X = np.array(data)
y = np.array(labels)


def bulid(X_train, y_train, batch_size=64, epochs=100):
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
                               activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape=(18, 201)),
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