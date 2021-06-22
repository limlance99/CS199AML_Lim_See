import torch

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
print(f"l2norms: {torch.mean(l2norms, 1, True).sum(1).mean()}")
print(f"success: {torch.mean(success, 1, True).sum(1).mean()}")