import torch

# stats = torch.load("results/pca_stats_1m.pth", map_location="cpu")
stats = torch.load("results/sub_pca.pth", map_location="cpu")

print("num_tokens:", stats["meta"]["num_tokens"])
print("dim:", stats["meta"]["dim"])

var = stats["var"]
print("top-10 var:", var[:10])
print("median var:", var.median())
print("min var:", var.min())
