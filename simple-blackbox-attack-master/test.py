def get_labels(filepath):
    labels = []
    with open(filepath) as f:
        labels = [line.strip().split(",") for line in f.readlines()]

    return labels

x = 'imagenet_classes.txt'
y = get_labels(x)
print(y[569])

