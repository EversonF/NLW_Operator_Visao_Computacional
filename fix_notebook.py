import json

nb_path = r'e:\Everson\NLW\NLW_Operator_Visao_Computacional\lenet5.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Verifica se a celula de instalacao ja existe
has_install = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('%pip install' in line for line in cell['source']):
        has_install = True
        break

if not has_install:
    new_cells = []
    for cell in nb['cells']:
        # Quando encontrar a primeira celula q usa matplotlib
        if cell['cell_type'] == 'code' and any('import matplotlib' in line for line in cell['source']):
            install_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Instalação das bibliotecas necessárias no kernel atual (caso não existam)\n", "%pip install matplotlib torchvision"]
            }
            new_cells.append(install_cell)
        new_cells.append(cell)
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with %pip install.")
else:
    print("Notebook already has %pip install.")
