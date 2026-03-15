import json

nb_path = r'e:\Everson\NLW\NLW_Operator_Visao_Computacional\lenet5.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Markdown section for clarity
save_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Salvando e Carregando o Modelo\n",
        "\n",
        "Uma vez que o modelo foi treinado e apresentou uma boa precisão, podemos salvar seus pesos em um arquivo `.pth`. Isso evita que precisemos treinar a rede do zero toda vez que quisermos utilizá-la para classificar novos dígitos numéricos."
    ]
}

# Code cell for saving the model state_dict
save_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "\n",
        "PATH = './lenet5_mnist_model.pth'\n",
        "\n",
        "# É uma boa prática salvar apenas o 'state_dict' (que contém os pesos aprendidos), em vez de todo o objeto modelo.\n",
        "torch.save(model.state_dict(), PATH)\n",
        "\n",
        "print(f\"Pesos do modelo salvos com sucesso no arquivo local: {os.path.abspath(PATH)}\")\n",
        "\n",
        "# ------------- OPCIONAL: Como carregar o modelo no futuro -------------\n",
        "# model_loaded = LeNet5() # Precisamos instanciar a estrutura do modelo novamente\n",
        "# model_loaded.load_state_dict(torch.load(PATH))\n",
        "# model_loaded.eval() # Definimos para modo de avaliação antes da inferência\n",
        "# print(\"Modelo recarregado do disco com sucesso!\")"
    ]
}

# Append the save sections
has_save_cell = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('lenet5_mnist_model.pth' in line for line in cell['source']):
        has_save_cell = True
        break

if not has_save_cell:
    nb['cells'].append(save_markdown_cell)
    nb['cells'].append(save_code_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with save model cells.")
else:
    print("Notebook already has the saving model cells.")
