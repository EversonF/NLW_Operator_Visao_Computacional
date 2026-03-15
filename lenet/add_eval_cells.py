import json

nb_path = r'e:\Everson\NLW\NLW_Operator_Visao_Computacional\lenet5.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Markdown section for clarity
eval_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Avaliação do Modelo no Conjunto de Teste\n",
        "\n",
        "Após treinar o modelo, precisamos avaliar sua performance utilizando dados que ele nunca viu duranto o treinamento. Para isso, vamos carregar o conjunto de testes (Test Dataset) do MNIST e calcular a precisão (accuracy) global da nossa rede neural."
    ]
}

# Code cell for loading test and evaluating
eval_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Carregando o dataset de teste do MNIST\n",
        "test_dataset = torchvision.datasets.MNIST(root='./data', \n",
        "                                          train=False, \n",
        "                                          transform=transform, \n",
        "                                          download=True)\n",
        "\n",
        "# DataLoader para os dados de teste\n",
        "test_loader = torch.utils.data.DataLoader(dataset=test_dataset, \n",
        "                                          batch_size=64, \n",
        "                                          shuffle=False)\n",
        "\n",
        "correct = 0\n",
        "total = 0\n",
        "\n",
        "# Como estamos apenas avaliando, não precisamos calcular gradientes\n",
        "with torch.no_grad():\n",
        "    for data in test_loader:\n",
        "        images, labels = data[0].to(device), data[1].to(device)\n",
        "        \n",
        "        # Calcula as predições da rede\n",
        "        outputs = model(images)\n",
        "        \n",
        "        # A classe predita é a que tem a maior energia\n",
        "        _, predicted = torch.max(outputs.data, 1)\n",
        "        \n",
        "        total += labels.size(0)\n",
        "        correct += (predicted == labels).sum().item()\n",
        "\n",
        "print(f'Precisão (Accuracy) da rede neural nas 10000 imagens de teste: {100 * correct / total} %')"
    ]
}

# Code cell for evaluating per class
eval_per_class_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Preparar os contadores para cada classe (dígitos 0 a 9)\n",
        "classes = [str(i) for i in range(10)]\n",
        "correct_pred = {classname: 0 for classname in classes}\n",
        "total_pred = {classname: 0 for classname in classes}\n",
        "\n",
        "with torch.no_grad():\n",
        "    for data in test_loader:\n",
        "        images, labels = data[0].to(device), data[1].to(device)\n",
        "        outputs = model(images)\n",
        "        _, predictions = torch.max(outputs, 1)\n",
        "        # Agrupar as predições corretas para cada classe\n",
        "        for label, prediction in zip(labels, predictions):\n",
        "            if label == prediction:\n",
        "                correct_pred[classes[label.item()]] += 1\n",
        "            total_pred[classes[label.item()]] += 1\n",
        "\n",
        "# Imprimir a precisão por classe\n",
        "for classname, correct_count in correct_pred.items():\n",
        "    accuracy = 100 * float(correct_count) / total_pred[classname]\n",
        "    print(f'Precisão para classe {classname:5s} is: {accuracy:.1f} %')"
    ]
}

# Append the eval sections
has_eval_cell = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('train=False' in line and 'torchvision.datasets.MNIST' in line for line in cell['source']):
        has_eval_cell = True
        break

if not has_eval_cell:
    nb['cells'].append(eval_markdown_cell)
    nb['cells'].append(eval_code_cell)
    nb['cells'].append(eval_per_class_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with evaluation cells.")
else:
    print("Notebook already has the evaluation cells.")
