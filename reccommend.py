# try running this to check if your PYTORCH uses GPU or not. Ideally GPU accelerated pytorch is good for such things
import torch
print(torch.cuda.is_available())