import torch


class EarlyStopping:
    def __init__(self, patience, delta, verbose=True, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None  # 存储准确率（越高越好）
        self.early_stop = False
        self.val_acc_max = -float('inf')  # 初始化为负无穷
        self.delta = delta
        self.path = path

    def __call__(self, val_acc, model):
        score = val_acc  # 准确率越高越好，无需取负

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:  # 准确率未提升
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # 准确率提升
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    # 保存模型
    def save_checkpoint(self, val_acc, model):
        if self.verbose:
            print(f'验证准确率提升 ({self.val_acc_max:.6f} --> {val_acc:.6f}). 保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc
