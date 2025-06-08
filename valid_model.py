if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from torchvision import models, datasets, transforms
    from torchvision.models import ResNet50_Weights
    import torch.nn as nn
    from torch import optim
    import time

    # 設定設備 (使用 GPU 加速)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加載模型
    # 載入 ResNet50 預訓練模型
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # 替換 ResNet 的分類層 (fc) 為 13 類
    num_classes = 13  # 假設你之前的模型有 13 類
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 加載已儲存的模型權重
    model.load_state_dict(torch.load("resnet50_model.pth"))
    model = model.to(device)

    # 2. 設定資料增強（轉換）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # 3. 載入驗證集
    val_dir = "validationset"  # 驗證集資料夾
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32,
                            shuffle=False, num_workers=2)

    # 4. 設置模型為評估模式
    model.eval()

    # 5. 驗證過程
    val_loss, val_correct = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    print("--start validing--")
    start = time.time()
    with torch.no_grad():  # 在驗證過程中禁用梯度計算，提高效率
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # 模型推理
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 累計損失和準確率
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    # 計算平均損失和準確率
    val_loss /= len(val_dataset)
    val_acc = val_correct.double() / len(val_dataset)
    end = time.time()
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, {end-start:.2f}S")
