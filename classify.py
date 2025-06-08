from sklearn.model_selection import train_test_split
import os
import shutil
import random
import pandas as pd

# 設定路徑
images_dir = 'images'  # 存放圖片的資料夾
train_csv = 'train.csv'  # 存放訓練集資料的 CSV
test_csv = 'test.csv'  # 存放測試集資料的 CSV
train_dir = 'trainset'  # 訓練集資料夾
test_dir = 'testset'  # 測試集資料夾
val_dir = 'validationset'  # 驗證集資料夾

# 建立訓練集與測試集資料夾（如果不存在）
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 讀取CSV檔案
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# 從CSV中取得id欄位
train_ids = train_df['id'].astype(str)
test_ids = test_df['id'].astype(str)

# 取得所有圖片檔案
all_images = os.listdir(images_dir)

# 記錄成功移動的圖片數量
move_train = 0
move_test = 0

# 依照id分類並移動圖片
for image in all_images:
    image_id = os.path.splitext(image)[0]  # 圖片名稱的id部分是檔名的前半部

    # 調試：查看每張圖片的id部分
    #print(f"處理圖片: {image}, ID: {image_id}")

    if image_id in train_ids.values:
        shutil.move(os.path.join(images_dir, image),
                    os.path.join(train_dir, image))
        move_train += 1
    elif image_id in test_ids.values:
        shutil.move(os.path.join(images_dir, image),
                    os.path.join(test_dir, image))
        move_test += 1

# 調試：輸出移動結果
print(f"成功將 {move_train} 張圖片移動到 trainset.")
print(f"成功將 {move_test} 張圖片移動到 testset.")

train_df = pd.read_csv(train_csv, dtype={'id': str, 'class': str})
os.makedirs(val_dir, exist_ok=True)
all_images = os.listdir(train_dir)

val_images = random.sample(all_images, int(len(all_images) * 0.2))
moved_to_val = 0

# 將選中的 20% 圖片移動到驗證集資料夾
for image in val_images:
    shutil.move(os.path.join(train_dir, image),
                os.path.join(val_dir, image))
    moved_to_val += 1

print(f"成功將 {moved_to_val} 張圖片從 trainset 移動到 validationset.")

# 取得剩餘的 80% 圖片
remaining_images = [
    image for image in all_images if image not in val_images]

# 根據 class 分類並移動圖片
for image in remaining_images:
    # 取得圖片的 ID（假設圖片的 ID 是去掉副檔名的檔名）
    image_id = os.path.splitext(image)[0]

    # 確保 id 存在於 train.csv，否則跳過
    class_info = train_df.loc[train_df['id'] == image_id, 'class']
    if class_info.empty:
        print(f"警告：找不到 {image_id} 在 train.csv 中，跳過此圖片。")
        continue  # 若找不到對應 class，則跳過該圖片

    # 取得該圖片的 class
    image_class = class_info.values[0]

    # 設定每個 class 的資料夾
    class_dir = os.path.join(train_dir, str(image_class))

    # 如果該 class 資料夾不存在，則創建它
    os.makedirs(class_dir, exist_ok=True)

    # 移動圖片到對應的 class 資料夾
    shutil.move(os.path.join(train_dir, image),
                os.path.join(class_dir, image))

print("train圖片已根據 class 分類並移動。")

for image in val_images:
    # 取得圖片的 ID（假設圖片的 ID 是去掉副檔名的檔名）
    image_id = os.path.splitext(image)[0]

    # 確保 id 存在於 train.csv，否則跳過
    class_info = train_df.loc[train_df['id'] == image_id, 'class']
    if class_info.empty:
        print(f"警告：找不到 {image_id} 在 train.csv 中，跳過此圖片。")
        continue  # 若找不到對應 class，則跳過該圖片

    # 取得該圖片的 class
    image_class = class_info.values[0]

    # 設定每個 class 的資料夾
    class_dir = os.path.join(val_dir, str(image_class))

    # 如果該 class 資料夾不存在，則創建它
    os.makedirs(class_dir, exist_ok=True)

    # 移動圖片到對應的 class 資料夾
    shutil.move(os.path.join(val_dir, image),
                os.path.join(class_dir, image))

print("validation圖片已根據 class 分類並移動。")
