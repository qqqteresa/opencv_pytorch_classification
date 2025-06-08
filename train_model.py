if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    import os
    import time
    torch.cuda.empty_cache()

    # 設定設備 (使用 GPU 加速)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 設定資料集路徑
    train_dir = "trainset"  # 訓練集資料夾
    val_dir = "validationset"  # 驗證集資料夾

    # 設定影像轉換（數據增強）
    transform = {
        'train': transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),  # 50% 機率水平翻轉
            transforms.RandomRotation(30),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ]),
    }

    # 載入數據集
    train_dataset = datasets.ImageFolder(
        root=train_dir, transform=transform['train'])


    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, num_workers=2)

    # 取得類別數量
    num_classes = len(train_dataset.classes)
    # print(f"Detected {num_classes} classes: {train_dataset.classes}")

    # 載入 ResNet50 預訓練模型 包含預訓練的權重
    # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 替換 ResNet 的分類層 (fc) 為 13 類
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 2. 加載儲存的模型權重
    model.load_state_dict(torch.load("resnet50_model.pth"))

    # 移動到 GPU
    model = model.to(device)

    # 設定損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 設定訓練 epoch
    num_epochs = 10

    # 訓練模型
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        # 訓練
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        train_loss /= len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        epoch_time = time.time() - start_time

        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},time: {epoch_time:.2f}s")

    # 儲存模型
    torch.save(model.state_dict(), "resnet50_model.pth")
    print("✅ Model saved as resnet50_model.pth")
