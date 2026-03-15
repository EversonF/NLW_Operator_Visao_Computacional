import json
import os

nb_path = r'e:\Everson\NLW\NLW_Operator_Visao_Computacional\lenet5.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Markdown section for clarity
load_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Carregando o Modelo e Fazendo Inferências Soltas\n",
        "\n",
        "Nesta etapa extra, vamos instanciar estruturalmente uma nova LeNet-5, carregar nela os pesos que salvamos (`.pth`), e solicitar que esta inteligência artificial faça uma classificação de um único dígito (para provar que ela realmente aprendeu e pode ser usada em produção)."
    ]
}

# Code cell for loading the model and doing inference
load_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import random\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# 1. Instanciamos uma rede 'limpa' do zero\n",
        "loaded_model = LeNet5()\n",
        "\n",
        "# 2. Carregamos o 'cérebro' do disco na rede (temos q definir onde mapear baseado no hw atual)\n",
        "PATH = './lenet5_mnist_model.pth'\n",
        "loaded_model.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))\n",
        "loaded_model.to(device)\n",
        "\n",
        "# 3. Definimos ela como modo inferência (Evaluation Mode)\n",
        "# Isso desliga Dropout ou BatchNorm caso a arquitetura tivesse estes itens\n",
        "loaded_model.eval()\n",
        "\n",
        "# 4. Vamos pegar uma IMAGEM ISOLADA para ela prever\n",
        "# Extraindo uma imagem aleatória do Test\n",
        "idx = random.randint(0, len(test_dataset) - 1)\n",
        "img, label = test_dataset[idx]\n",
        "\n",
        "# O PyTorch espera um Batch (Lote). Então fazemos img.unsqueeze(0)\n",
        "# para transformar formato [1, 32, 32] em [1, 1, 32, 32] (Lote de tamanho 1)\n",
        "img_batch = img.unsqueeze(0).to(device)\n",
        "\n",
        "with torch.no_grad():\n",
        "    output = loaded_model(img_batch)\n",
        "    \n",
        "    # Pega o valor máximo (probabilidade da classe)\n",
        "    _, predicted = torch.max(output, 1)\n",
        "\n",
        "# Desnormalizando para plotar com matplotlib\n",
        "plt_img = img[0].numpy() * 0.5 + 0.5\n",
        "plt.imshow(plt_img, cmap='gray')\n",
        "plt.title(f\"Real: {label} | A IA previu: {predicted.item()}\")\n",
        "plt.axis('off')\n",
        "plt.show()"
    ]
}

# Append the load sections
has_load_cell = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('loaded_model.load_state_dict' in line for line in cell['source']):
        has_load_cell = True
        break

if not has_load_cell:
    nb['cells'].append(load_markdown_cell)
    nb['cells'].append(load_code_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with load and inference cells.")
else:
    print("Notebook already has the loading model inference cells.")
