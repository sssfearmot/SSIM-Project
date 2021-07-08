# pytorch-ssim for 3D images

Implementation of pytorch SSIM loss for 3D images based on [Po-Hsun-Su's Repository](https://github.com/Po-Hsun-Su/pytorch-ssim) 

## Example
### basic usage
```python
import pytorch_ssim
import torch
from torch.autograd import Variable

img1 = Variable(torch.rand(1, 1, 256, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256, 256))

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_ssim.ssim3D(img1, img2))

ssim_loss = pytorch_ssim.SSIM3D(window_size = 11)

print(ssim_loss(img1, img2))
