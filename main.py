from extract_subset import extract_subset
from split_modality import split_modality
from train import train_model
from test_model import test_model


def plot_loss_curve(model_id):
    import pandas as pd
    import matplotlib.pyplot as plt

    # 读取loss日志
    df = pd.read_csv(f"./saved_models/model_{model_id}/loss_logs/loss_log.csv")
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['loss'], label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Model {model_id} Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # extract_subset(
    #     num_samples=10000,   # 随机取10000个样本
    #     random_state=42      # 固定随机种子
    # )

    # split_modality(
    #     dataset_id=4,   # 对4号数据集进行模态分离
    #     random_state=42      # 固定随机种子
    # )

    # try:
    #     train_model(
    #         dataset_id=4,   # 对4号数据集进行训练
    #         split_id=0,   # 对0号模态分离进行训练
    #         per_device_train_batch_size=2,   # 每个设备训练批次大小
    #         gradient_accumulation_steps=4,   # 梯度累加步数
    #         learning_rate=1e-4,   # 学习率
    #         num_train_epochs=2,   # 训练轮数
    #     )
    # except Exception as e:
    #     print(f"出错了：{e}")
    
    # plot_loss_curve(1)
    
    try:
        result =test_model(
            dataset_id=4,   # 选择2号数据集进行测试
            split_id=0,   # 选择1号划分的测试集进行测试
            model_id=0,   # 选择0号训练参数进行测试
            verbose=True,   # 打印测试结果
        )    

        # for key,value in result.items():
        #     print(f"{key}: {value}")
    except Exception as e:
        print(f"出错了：{e}")





if __name__ == "__main__":
    main()