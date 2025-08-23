# inference.py独立测试文件
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import sys
from pathlib import Path


# ---------- 关键路径处理 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 向上回溯
sys.path.append(str(PROJECT_ROOT))


def load_class_names_from_folders(test_dir):
    """从测试集文件夹的子文件夹名加载类别名"""
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"测试集文件夹未找到: {test_dir}")

    # 获取所有子文件夹名并排序
    class_names = sorted([d for d in os.listdir(test_dir)
                          if os.path.isdir(os.path.join(test_dir, d))])

    if not class_names:
        raise ValueError("测试集文件夹中没有找到任何子文件夹/类别")

    print(f"\n>>从文件夹加载了 {len(class_names)} 个类别")
    return class_names


def load_onnx_model(onnx_path):
    """加载ONNX模型并创建推理会话"""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX模型文件未找到: {onnx_path}")

    # 创建ONNX Runtime推理会话，优先使用GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def preprocess_image(image_path):
    """预处理图像"""
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 固定尺寸
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image_path).convert('RGB')
    return inference_transform(img).unsqueeze(0)  # 添加batch维度


def predict_image(session, image_tensor, class_names):
    """执行推理预测并返回类别名和置信度"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 转换为numpy数组
    input_data = image_tensor.numpy()

    # 运行推理
    outputs = session.run([output_name], {input_name: input_data})[0]

    # 获取预测结果,Softmax归一化处理
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # 防止数值溢出
        return e_x / e_x.sum(axis=1, keepdims=True)

    probs = softmax(outputs)
    predicted_idx = np.argmax(probs)
    confidence = np.max(probs)

    # 如果置信度低于阈值，则返回未知类别
    if confidence < 0.3:  # 置信阈值
        class_name = "未知类别"
    else:
        class_name = class_names[predicted_idx]

    return class_name, confidence


def main():

    # 配置路径
    ONNX_MODEL_PATH = str(PROJECT_ROOT / "model" / "final_model.onnx")  # ONNX模型路径
    TEST_DIR = str(PROJECT_ROOT / "dataset" / "cards_image"/ "img" / "test")  # 拼接测试集路径
    IMAGE_PATH = str(PROJECT_ROOT / IMAGE)  # 预测图像路径

    print("*"*50 + "\n开始进行推理\n" + "*"*50)

    try:
        # 1. 从测试集文件夹加载类别名
        class_names = load_class_names_from_folders(TEST_DIR)

        # 2. 加载模型
        session = load_onnx_model(ONNX_MODEL_PATH)
        print(">>ONNX模型加载成功")

        # 3. 预处理图像
        if not os.path.exists(IMAGE_PATH):
            raise FileNotFoundError(f"要预测的图片未找到: {IMAGE_PATH}")

        input_tensor = preprocess_image(IMAGE_PATH)
        print(f">>图像预处理完成")

        # 4. 执行推理
        class_name, confidence = predict_image(session, input_tensor, class_names)

        # 5. 打印结果
        print("\n>>预测结果如下")
        # print(f"图片路径: {IMAGE_PATH}")
        print(f"预测类别: {class_name}")
        print(f"置信度: {confidence:.2%}")

    except Exception as e:
        print(f"推理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    # IMAGE = "image_pred/Spades_A.png"
    # IMAGE = "image_pred/Plum_K.jpg"
    IMAGE = "image_pred/joker.jpg"
    main()
