if __name__ == '__main__':
    import pandas as pd
    import torch
    from torchvision import transforms
    import torchvision.models as models
    from PIL import Image
    from torch.utils.data import DataLoader
    import os
    import torch.nn as nn

    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 載入你的模型
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 重新設定全連接層，使其輸出 13 個類別
    num_classes = 13  # 記得改成你模型的類別數
    # 替換 ResNet 的分類層 (fc) 為 13 類
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 將模型載入到 GPU/CPU
    model.load_state_dict(torch.load("resnet50_model.pth", map_location=device))
    model.to(device)
    model.eval()  # 設定為推論模式

    # 測試圖片的資料夾（Kaggle 通常會給你一個 `test/` 資料夾）
    test_folder = "testset"  # 你的測試圖片資料夾
    image_paths = [os.path.join(test_folder, img)
                for img in os.listdir(test_folder)]

    # 數據增強（要與訓練時相同）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 確保大小與模型匹配
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定義類別名稱（根據你的模型訓練的類別順序）
    class_names = ["bhaji", "chapati", "githeri", "kachumbari",
                   "kukuchoma", "mandazi", "masalachips", "matoke", "mukimo", "nyamachoma", "pilau", "sukumawiki", "ugali"]

    # 儲存預測結果
    results = []

    # 進行預測
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")  # 讀取圖片
            image = transform(image).unsqueeze(0).to(device)  # 預處理並轉為 batch
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]  # 取得類別名稱
            image_id = os.path.splitext(os.path.basename(image_path))[
                0]  # 取得圖片 ID（去掉副檔名）

            results.append({"id": image_id, "class": predicted_class})

    # 轉換為 DataFrame
    df = pd.DataFrame(results)

    # 儲存為 CSV
    df.to_csv("submission.csv", index=False)

    print("已成功生成 Kaggle 提交檔案：submission.csv")
