import json
import os

nb_path = r'e:\Everson\NLW\NLW_Operator_Visao_Computacional\lenet5.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Markdown section for clarity
fm_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Visualizando os Mapas de Características (Feature Maps)\n",
        "\n",
        "Uma das grandes vantagens das Redes Neurais Convolucionais é que podemos \"espiar\" o que a rede está vendo internamente. Abaixo, pegamos a imagem anterior e a passamos apenas pela primeira camada convolucional (`conv1`), extraindo os 6 tensores resultantes e exibindo-os. Estes são os filtros de borda e padrões ativados!"
    ]
}

# Code cell for generating and plotting feature maps
fm_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import torch.nn.functional as F\n",
        "\n",
        "# Vamos pegar a imagem (img_batch) que instanciamos no bloco anterior\n",
        "with torch.no_grad():\n",
        "    # Em vez de chamar o forward() completo, vamos passar o dado apenas pela conv1 e relu\n",
        "    conv1_out = F.relu(loaded_model.conv1(img_batch))\n",
        "\n",
        "# O Output da conv1 tem formato [Lote, 6_canais, 28_altura, 28_largura]\n",
        "feature_maps = conv1_out[0].cpu().numpy()\n",
        "\n",
        "# Configurando o Plot (1 linha, 6 colunas para os 6 filtros da LeNet-5 conv1)\n",
        "fig, axes = plt.subplots(1, 6, figsize=(16, 3))\n",
        "\n",
        "for i in range(6):\n",
        "    ax = axes[i]\n",
        "    # Mostrando cada um dos 6 canais\n",
        "    ax.imshow(feature_maps[i], cmap='viridis') # Viridis é ótimo para ver intensidade de ativação\n",
        "    ax.set_title(f'Mapa de Ativação {i+1}')\n",
        "    ax.axis('off')\n",
        "\n",
        "plt.suptitle(f'O que a 1ª Camada da Rede Vê na imagem do dígito {label}', fontsize=16)\n",
        "plt.show()"
    ]
}

# Append the feature map sections
has_fm_cell = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('feature_maps = conv1_out' in line for line in cell['source']):
        has_fm_cell = True
        break

if not has_fm_cell:
    nb['cells'].append(fm_markdown_cell)
    nb['cells'].append(fm_code_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with feature map visualization.")
else:
    print("Notebook already has the feature map cells.")
