import torch
import torchvision.datasets as dset
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import random
import utils


# testset = dset.ImageFolder('datasets/googledimages', utils.IMAGENET_TRANSFORM)
# testing = testset[random.randint(0, len(testset) - 1)]
# image = testing[0]
# label = testset.classes[testing[1]]
# print(label, type(label))
real_labels = utils.get_labels("imagenet_classes.txt")
x = torch.load("save\\pixel_resnet50_500_10000_224_0.2000_randgoogledimages.pth")
# all_the_shit = f'''
# adv: {x["adv"]}
# probs: {x["probs"]}
# succs: {x["succs"]}
# queries: {x["queries"]}
# l2_norms: {x["l2_norms"]}
# linf_norms: {x["linf_norms"]}
# 	'''
queries = x["queries"]
l2norms = x["l2_norms"]
success = x["succs"]

print(f"queries: {queries.sum(1).mean()}")
print(f"l2norms: {torch.mean(l2norms, 1, True).mean()}")
print(f"success: {torch.mean(success, 1, True).mean()}")

batchfile = 'googledimages/images_resnet50_500.pth'
checkpoint = torch.load(batchfile)
images = checkpoint['images']
labels = checkpoint['labels']
adversarials = x["adv"]

# test_num = 3
# model = models.resnet50(pretrained=True)
# model.eval()
# batch_t = torch.unsqueeze(images[test_num], 0)
# out = model(batch_t)
# _, index = torch.max(out, 1)
# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# _, indices = torch.sort(out, descending=True)
# print([(idx, percentage[idx].item()) for idx in indices[0][:5]])

# model.eval()
# batch_t = torch.unsqueeze(adversarials[test_num], 0)
# out = model(batch_t)
# _, index = torch.max(out, 1)
# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# _, indices = torch.sort(out, descending=True)
# print([(idx, percentage[idx].item()) for idx in indices[0][:5]])

width=10
height=10
rows = 5
cols = 2
axes=[]
fig=plt.figure(figsize=(10,10))
plt.axis("off")

for a in range(5):
	origs = [i for i in range(1,10,2)]  # 1 3 5 7 9
	advs = [i for i in range(2,11,2)] # 2 4 6 8  
	randnum = random.randint(0, 499)
	axes.append( fig.add_subplot(rows, cols, origs[a]) )
	subplot_title=(f"(Orig) {real_labels[labels[randnum]]}")
	axes[-1].set_title(subplot_title)
	plt.imshow(images[randnum].permute(1,2,0))

	axes.append( fig.add_subplot(rows, cols, advs[a]) )
	subplot_title=(f"(Adv) {real_labels[labels[randnum]]}")
	axes[-1].set_title(subplot_title)  
	plt.imshow(adversarials[randnum].permute(1,2,0))
fig.tight_layout()    
plt.show()