import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# Define the export directory
current_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(current_dir, 'export')
# Load prepared data
fcbs_aspects_elements = torch.load(os.path.join(export_dir, 'fcbs_aspects_elements.pt'))
fcbs_build_ups_details = torch.load(os.path.join(export_dir, 'fcbs_build_ups_details.pt'))
fcbs_sectors_subsectors = torch.load(os.path.join(export_dir, 'fcbs_sectors_subsectors.pt'))
ice_db = torch.load(os.path.join(export_dir, 'ice_db.pt'))
clf_embodied_carbon = torch.load(os.path.join(export_dir, 'clf_embodied_carbon.pt'))

# Extracting numerical and tokenized features
def extract_features(df, numerical_cols, tokenized_cols):
    numerical_features = df[numerical_cols].values
    tokenized_features = np.array(list(df[tokenized_cols[0]]))  # Assuming single tokenized column for simplicity
    return torch.tensor(np.hstack([tokenized_features, numerical_features]), dtype=torch.float32)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# First-Stage Regression: ICE DB
X_ice = extract_features(ice_db, ['Carbon Storage Amount (Kg CO2 per unit)', 'Embodied Carbon per kg (kg CO2e per kg)'], ['Material'])
y_ice = torch.tensor(ice_db['Embodied Carbon (kg CO2e per declared unit)'].values, dtype=torch.float32).unsqueeze(1)

ice_dataset = TensorDataset(X_ice, y_ice)
ice_loader = DataLoader(ice_dataset, batch_size=32, shuffle=True)

input_size = X_ice.shape[1]
ice_model = SimpleNN(input_size)

# Training the ICE model
criterion = nn.MSELoss()
optimizer = optim.Adam(ice_model.parameters(), lr=0.001)

for epoch in range(100):  # Number of epochs
    for inputs, targets in ice_loader:
        optimizer.zero_grad()
        outputs = ice_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

ice_effects = ice_model(X_ice).detach().numpy()

# Save the ICE model
torch.save(ice_model.state_dict(), os.path.join(export_dir, 'ice_model.pth'))

# Adding ICE effects to aspects and build-ups
fcbs_aspects_elements['ice_effects'] = np.mean(ice_effects)
fcbs_build_ups_details['ice_effects'] = np.mean(ice_effects)

# Second-Stage Regression: FCBS Aspects and Build-ups
X_aspects = extract_features(fcbs_aspects_elements, ['Mass (kg/unit)', 'Weight (kN/unit)', 'Sequestered Carbon (kgCO2e/unit)', 'ice_effects'], ['Element'])
y_aspects = torch.tensor(fcbs_aspects_elements['Embodied carbon (kgCO2e/unit)'].values, dtype=torch.float32).unsqueeze(1)

aspects_dataset = TensorDataset(X_aspects, y_aspects)
aspects_loader = DataLoader(aspects_dataset, batch_size=32, shuffle=True)

input_size = X_aspects.shape[1]
aspect_model = SimpleNN(input_size)

# Training the Aspects model
optimizer = optim.Adam(aspect_model.parameters(), lr=0.001)

for epoch in range(100):
    for inputs, targets in aspects_loader:
        optimizer.zero_grad()
        outputs = aspect_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

aspect_effects = aspect_model(X_aspects).detach().numpy()

# Save the Aspects model
torch.save(aspect_model.state_dict(), os.path.join(export_dir, 'aspect_model.pth'))

X_buildups = extract_features(fcbs_build_ups_details, ['TOTAL BIOGENIC kgCO2e/FU', 'ice_effects'], ['Build-up Reference'])
y_buildups = torch.tensor(fcbs_build_ups_details['TOTAL kgCO2e/FU'].values, dtype=torch.float32).unsqueeze(1)

buildups_dataset = TensorDataset(X_buildups, y_buildups)
buildups_loader = DataLoader(buildups_dataset, batch_size=32, shuffle=True)

input_size = X_buildups.shape[1]
buildup_model = SimpleNN(input_size)

# Training the Buildups model
optimizer = optim.Adam(buildup_model.parameters(), lr=0.001)

for epoch in range(100):
    for inputs, targets in buildups_loader:
        optimizer.zero_grad()
        outputs = buildup_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

buildup_effects = buildup_model(X_buildups).detach().numpy()

# Save the Buildups model
torch.save(buildup_model.state_dict(), os.path.join(export_dir, 'buildup_model.pth'))

# Adding aspect and buildup effects to sectors
fcbs_sectors_subsectors['aspect_effects'] = np.mean(aspect_effects)
fcbs_sectors_subsectors['buildup_effects'] = np.mean(buildup_effects)

# Third-Stage Regression: FCBS Sectors/Subsectors
X_sectors = extract_features(fcbs_sectors_subsectors, ['Grid size:', 'Typical (Electric)', 'Best Practice (Electric)', 'Innovative (Electric)', 
                                                      'Typical (Non-electric)', 'Best Practice (Non-electric)', 'Innovative (Non-electric)',
                                                      'Typical (Total)', 'Best Practice (Total)', 'Innovative (Total)', 'Max (Total)',
                                                      'aspect_effects', 'buildup_effects'], ['Sector'])
y_sectors = torch.tensor(fcbs_sectors_subsectors['Typical (Total)'].values, dtype=torch.float32).unsqueeze(1)

sectors_dataset = TensorDataset(X_sectors, y_sectors)
sectors_loader = DataLoader(sectors_dataset, batch_size=32, shuffle=True)

input_size = X_sectors.shape[1]
sector_model = SimpleNN(input_size)

# Training the Sectors model
optimizer = optim.Adam(sector_model.parameters(), lr=0.001)

for epoch in range(100):
    for inputs, targets in sectors_loader:
        optimizer.zero_grad()
        outputs = sector_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

sector_effects = sector_model(X_sectors).detach().numpy()

# Save the Sectors model
torch.save(sector_model.state_dict(), os.path.join(export_dir, 'sector_model.pth'))

# Adding sector effects to final dataset
clf_embodied_carbon['sector_effects'] = np.mean(sector_effects)

# Final-Stage Regression: CLF Embodied Carbon
X_clf = extract_features(clf_embodied_carbon, ['Embodied Carbon Life Cycle Assessment Area Per Square Meter', 'Minimum Building Area in Square Meters', 
                                               'Maximum Building Area in Square Meters', 'Minimum Building Area in Square Feet', 
                                               'Maximum Building Area in Square Feet', 'Minimum Building Storeys', 'Maximum Building Storeys',
                                               'sector_effects'], ['Building Type'])
y_clf = torch.tensor(clf_embodied_carbon['Embodied Carbon Whole Building Excluding Operational'].values, dtype=torch.float32).unsqueeze(1)

clf_dataset = TensorDataset(X_clf, y_clf)
clf_loader = DataLoader(clf_dataset, batch_size=32, shuffle=True)

input_size = X_clf.shape[1]
clf_model = SimpleNN(input_size)

# Training the CLF model
optimizer = optim.Adam(clf_model.parameters(), lr=0.001)

for epoch in range(100):
    for inputs, targets in clf_loader:
        optimizer.zero_grad()
        outputs = clf_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

clf_predictions = clf_model(X_clf).detach().numpy()

# Save the CLF model
torch.save(clf_model.state_dict(), os.path.join(export_dir, 'clf_model.pth'))

# Evaluation
final_mse = mean_squared_error(y_clf.numpy(), clf_predictions)
print(f'Final Mean Squared Error: {final_mse}')

# Export the final model predictions
clf_embodied_carbon['Final_Predicted_Embodied_Carbon'] = clf_predictions
clf_embodied_carbon.to_csv(os.path.join(export_dir, 'final_predictions.csv'), index=False)

print("Model training completed and predictions saved.")
