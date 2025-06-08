import torch

# 檢查 CUDA 是否可用
print("CUDA Available:", torch.cuda.is_available())

# 如果 CUDA 可用，顯示 GPU 名稱
print("GPU Name:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "No GPU")

# 顯示 PyTorch 支持的 CUDA 版本
print("CUDA Version:", torch.version.cuda)

# 顯示 cuDNN 版本
print("cuDNN Version:", torch.backends.cudnn.version()
      if torch.backends.cudnn.is_available() else "No cuDNN")

# 設定設備 (使用 GPU 加速)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
