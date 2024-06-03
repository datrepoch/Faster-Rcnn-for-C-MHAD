
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import torch
import torchvision
from torchvision.transforms import functional as F

# 定义数据集的路径
DATASET_PATH = "dataset"

# 加载数据函数
def load_dataset():
    print("Loading dataset...")
    # 假设标签文件名为'labels.csv'，并且存储在数据集根目录中
    labels = pd.read_csv(os.path.join(DATASET_PATH, 'labels.csv'))
    
    all_data = []
    all_labels = []
    
    # 设置Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    for index, row in labels.iterrows():
        activity_label = row['activity']
        subject_id = row['subject']
        trial_id = row['trial']
        
        file_path = os.path.join(DATASET_PATH, 'videos', f"subject_{subject_id}", f"video_sub{subject_id}_tr{trial_id}.avi")
        
        try:
            cap = cv2.VideoCapture(file_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 转换为PIL图像
                pil_frame = F.to_pil_image(frame)
                # 转换为张量
                tensor_frame = F.to_tensor(pil_frame).unsqueeze(0)
                
                # 使用Faster R-CNN进行对象检测
                with torch.no_grad():
                    outputs = model(tensor_frame)
                
                detected_frame = frame  # 在这里你可以处理检测结果
                
                # 将帧转换为灰度图像并缩放到固定大小
                gray_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(gray_frame, (64, 64))
                frames.append(resized_frame)
            cap.release()
            frames = np.array(frames)
            all_data.append(frames)
            all_labels.append(activity_label)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print("Dataset loaded.")
    return np.array(all_data), np.array(all_labels)

# 加载数据
X, y = load_dataset()
print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# 数据预处理
print("Preprocessing data...")
X = X.reshape((X.shape[0], X.shape[1], -1))  # 将每帧展平
y = to_categorical(y)
print(f"Data preprocessed: {X.shape}, Labels shape: {y.shape}")

# 划分训练和测试集
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# 数据标准化
print("Standardizing data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
print("Data standardized.")

# 构建模型
print("Building model...")
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model built.")

# 定义模型保存路径和回调（保存路径自行设置）
checkpoint = ModelCheckpoint('D:/best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

# 训练模型时使用回调
print("Starting model training...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
print("Model training completed.")

# 评估模型（加载最佳模型）
print("Evaluating model...")
model.load_weights('D:/best_model.keras')
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 保存最终模型
print("Saving final model...")
model.save('D:/final_model.keras') 
print("Final model saved.")
