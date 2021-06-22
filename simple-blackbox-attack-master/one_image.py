import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torchvision.datasets as dset

from simba import SimBA

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
args = parser.parse_args()

model = getattr(models, args.model)(pretrained=True).cuda()
model.eval()
if args.model.startswith('inception'):
    image_size = 299
else:
    image_size = 224
attacker = SimBA(model, 'imagenet', image_size)


batchfile = 'save/images_resnet50_1000.pth'
checkpoint = torch.load(batchfile)
images = checkpoint['images']
labels = checkpoint['labels']
one_image = images[0]
one_label = labels[0]

print(one_label.numpy())

one_adv = attacker.simba_single(one_image, one_label)

fig = plt.figure()
rows = 1
cols = 2

fig.add_subplot(rows, cols, 1)
plt.imshow(one_image.permute(1,2,0))
plt.title("Original")

fig.add_subplot(rows, cols, 2)
plt.imshow(one_adv.permute(1,2,0))
plt.title("Adversarial")
plt.show()

