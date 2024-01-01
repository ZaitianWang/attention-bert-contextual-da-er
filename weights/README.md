# model weights

name format: `{score}-{day}-{hourminute}-{epoch}-{batch}.pt`

saved with model architecture and optimizer state with `torch.save(model, path)`
loaded with `torch.load(path)`
