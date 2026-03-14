import json

nb_path = r'e:\Everson\NLW\NLW_Operator_Visao_Computacional\lenet5.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell to visualize MNIST
mnist_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import torch\n",
        "import torchvision\n",
        "import torchvision.transforms as transforms\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "# Transform to normalize the data\n",
        "transform = transforms.Compose([\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize((0.5,), (0.5,))\n",
        "])\n",
        "\n",
        "# Download and load the training data\n",
        "trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)\n",
        "trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)\n",
        "\n",
        "# Function to show an image\n",
        "def imshow(img):\n",
        "    # unnormalize\n",
        "    img = img / 2 + 0.5\n",
        "    npimg = img.numpy()\n",
        "    # Since MNIST is grayscale, we can use cmap='gray' on the 2D array or let plot handle the 3D grid\n",
        "    plt.imshow(np.transpose(npimg, (1, 2, 0)))\n",
        "    plt.axis('off')\n",
        "    plt.show()\n",
        "\n",
        "# Get some random training images\n",
        "dataiter = iter(trainloader)\n",
        "images, labels = next(dataiter)\n",
        "\n",
        "# Show a grid containing 16 images\n",
        "imshow(torchvision.utils.make_grid(images[:16], nrow=4))\n",
        "# Print labels\n",
        "print('Labels:', ' '.join(f'{labels[j].item()}' for j in range(16)))\n"
    ]
}

# Add exactly one MNIST visual cell at the end if it doesn't already exist
has_mnist_cell = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('torchvision.datasets.MNIST' in line for line in cell['source']):
        has_mnist_cell = True
        break

if not has_mnist_cell:
    nb['cells'].append(mnist_cell)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with MNIST visualização cell.")
else:
    print("Notebook already has the MNIST cell.")
