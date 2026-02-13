import torch
print(torch.__version__)              # Should show torch-2.x+cu121
print(torch.cuda.is_available())      # Should be True
print(torch.cuda.get_device_name(0))  # Your GPU name
