import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


data = pd.read_csv('dataset/housing.csv')
data.dropna(inplace=True)

kolonka = pd.get_dummies(data["ocean_proximity"]).astype(int)
data = data.join(kolonka)
data = data.drop(["ocean_proximity"], axis=1)

input_data = data.drop(["median_house_value"], axis=1)
output_data = data["median_house_value"]


skala = StandardScaler()
input_data = skala.fit_transform(input_data)

scaler_output = MinMaxScaler()
output_data = scaler_output.fit_transform(output_data.values.reshape(-1, 1))


train_input, test_input, train_output, test_output = train_test_split(input_data, output_data, test_size=0.2)


# Prevod na tenzory
torch_train_input = torch.tensor(train_input).type(torch.float32)
torch_train_output = torch.tensor(train_output).type(torch.float32)

torch_test_input = torch.tensor(test_input).type(torch.float32)
torch_test_output = torch.tensor(test_output).type(torch.float32)

# Dataset a DataLoader
train_dataset = TensorDataset(torch_train_input, torch_train_output)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = TensorDataset(torch_test_input, torch_test_output)
test_loader = DataLoader(test_dataset, batch_size=128)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l_vrstva1 = nn.Linear(13, 64)
        self.l_vrstva2 = nn.Linear(64, 32)
        self.l_vrstva3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.tanh(self.l_vrstva1(x))
        x = F.relu(self.l_vrstva2(x))
        x = self.l_vrstva3(x)
        return x



def spustenie():
    mseloss = nn.MSELoss()
    train_chyba = []
    test_chyba = []

    # Tréning
    model.train()
    for epocha in range(200):
        priemer_chyba = 0
        for x_batch, y_batch in train_loader:
            optimal.zero_grad()
            output = model(x_batch)
            chyba = mseloss(output, y_batch)
            chyba.backward()
            optimal.step()
            priemer_chyba += chyba.item()

        priemer_chyba /= len(train_loader)
        train_chyba.append(priemer_chyba)

        model.eval()  # Prepnutie do evaluačného režimu
        priemer_chyba = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                output = model(x_batch)
                chyba = mseloss(output, y_batch)
                priemer_chyba += chyba.item()

        priemer_chyba /= len(test_loader)
        test_chyba.append(priemer_chyba)

        if epocha % 10 == 0 and epocha != 0:
            print(
                f"Epocha {epocha}: Chyba pri trenovani = {train_chyba[-1]:.4f}")

    # Testovanie
    model.eval()  # Prepnutie do evaluačného režimu
    cela_chyba = 0
    predikcie_output = []
    good_output = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            chyba = mseloss(output, y_batch)
            cela_chyba += chyba.item()
            predikcie_output.extend(output.numpy().flatten())
            good_output.extend(y_batch.numpy().flatten())

    # Výpočet RMSE
    cela_chyba /= len(test_loader)
    rmse = torch.sqrt(torch.tensor(cela_chyba))

    # Denormalizácia predpovedí a skutočných hodnôt na pôvodný rozsah
    predikcie_output = scaler_output.inverse_transform(np.array(predikcie_output).reshape(-1, 1)).flatten()
    good_output = scaler_output.inverse_transform(np.array(good_output).reshape(-1, 1)).flatten()

    # Výpočet percentuálnej chyby
    error = np.abs((good_output - predikcie_output) / good_output) * 100
    accuracy = 100 - np.mean(error)

    print("Test RMSE:", rmse.item())
    print(f"Uspesnost: {accuracy:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(train_chyba, label='Trénovacia chyba', color='blue')
    plt.plot(test_chyba, label='Testovacia chyba', color='red')
    plt.xlabel("Epochy")
    plt.ylabel("Chyba")
    plt.title("Priebeh trénovacej a testovacej chyby")
    plt.legend()
    plt.grid(True)
    plt.show()


model = Model()

a = int(input("1 - Adam\t2 - SGD bez momentum\t3 - SGD s momentum\nZadaj:"))
if a == 1:
    print("------------------------------ ADAM ------------------------------")
    optimal = torch.optim.Adam(model.parameters(), lr=0.01)
    spustenie()
elif a == 2:
    print("------------------ SGD bez momentum -------------------")
    optimal = torch.optim.SGD(model.parameters(), lr=0.01)
    spustenie()
elif a == 3:
    print("------------------ SGD s momentum --------------------")
    optimal = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    spustenie()

