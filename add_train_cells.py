import json

nb_path = r'e:\Everson\NLW\NLW_Operator_Visao_Computacional\lenet5.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Markdown section for clarity
train_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Treinamento do Modelo\n",
        "\n",
        "Agora vamos definir a função de custo (Cross Entropy Loss) e o otimizador (Adam ou SGD) para iniciar o treinamento da rede neural LeNet-5 com os dados de treinamento."
    ]
}

# Code cell for training loop
train_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import torch.optim as optim\n",
        "\n",
        "# Definimos a Fun\u00e7\u00e3o de Perda e o Otimizador\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
        "\n",
        "# Definir dispositivo de hardware (GPU ou CPU)\n",
        "device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(f\"Treinando no dispositivo: {device}\")\n",
        "model.to(device)\n",
        "\n",
        "# Loop de Treinamento\n",
        "epochs = 5\n",
        "print(\"Iniciando o treinamento...\")\n",
        "for epoch in range(epochs):  # loop over the dataset multiple times\n",
        "\n",
        "    running_loss = 0.0\n",
        "    for i, data in enumerate(train_loader, 0):\n",
        "        # Obter os inputs; data é uma lista de [inputs, labels]\n",
        "        inputs, labels = data[0].to(device), data[1].to(device)\n",
        "\n",
        "        # Zerar as derivadas/gradientes\n",
        "        optimizer.zero_grad()\n",
        "\n",
        "        # Forward (caminho de ida) + backward (caminho de volta para calcular os gradientes) + optimize (atualizar pesos)\n",
        "        outputs = model(inputs)\n",
        "        loss = criterion(outputs, labels)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        # Imprimir estátisticas no final de vários batchs\n",
        "        running_loss += loss.item()\n",
        "        if i % 2000 == 1999:    # printar a cada 2000 mini-batches\n",
        "            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')\n",
        "            running_loss = 0.0\n",
        "\n",
        "print('Treinamento Finalizado!')"
    ]
}

# Append the training section
has_train_cell = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('optim.Adam' in line or 'loss.backward()' in line for line in cell['source']):
        has_train_cell = True
        break

if not has_train_cell:
    nb['cells'].append(train_markdown_cell)
    nb['cells'].append(train_code_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with training loops.")
else:
    print("Notebook already has the training cell.")
