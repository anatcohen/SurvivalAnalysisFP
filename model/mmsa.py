import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from typing import Dict
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index as lifelines_cindex
from config.paths import DATA_DIR
from preprocessing.clinical_data.clinical_data_preprocessing import preprocess_data

import sys
import datetime

class SurvivalDataset(Dataset):
    """Dataset for multi-modal survival analysis."""

    def __init__(self, clinical_csv: str, processed_dir: str, mode='train',
                 transform=None, clinical_transform=None):
        """
        Args:
            clinical_csv: Path to clinical data CSV
            processed_dir: Directory with processed CT/mask files
            transform: Optional transform for CT data
            mode: train/val/test
            clinical_transform: Optional transform for clinical data
        """
        self.processed_dir = processed_dir
        self.transform = transform
        self.clinical_transform = clinical_transform
        self.mode = mode

        # Load clinical data
        self.clinical_df = pd.read_csv(clinical_csv)
        self.clinical_df.set_index('PatientID', inplace=True)

        # Cache available files for efficiency
        available_files = set(os.listdir(processed_dir))

        # Get list of processed files
        self.subject_ids = []
        for patient in self.clinical_df.index:
            if mode == 'train':
                for aug_id in range(1, 5):  # 4 augmentations per patient
                    fname = f"{patient}_{aug_id}.npz"
                    if fname in available_files:
                        subject_id = f"{patient}_{aug_id}"
                        self.subject_ids.append(subject_id)
            else:
                fname = f"{patient}_1.npz"  # only original non augmented is used
                if fname in available_files:
                    subject_id = f"{patient}_1"
                    self.subject_ids.append(subject_id)

        print(f"[{mode}] Found {len(self.subject_ids)} subjects with both imaging and clinical data")

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]

        # Load imaging data
        npz_path = os.path.join(self.processed_dir, f"{subject_id}.npz")
        data = np.load(npz_path)
        masked_ct = data['masked_ct']

        # Add channel dimension
        masked_ct = masked_ct[np.newaxis, ...]  # (1, D, H, W)

        # Apply transform if provided
        if self.transform:
            masked_ct = self.transform(masked_ct)

        subject_clinical_id = subject_id.rsplit('_', 1)[0]  # Augmented patients share clinical data
        clinical_row = self.clinical_df.loc[subject_clinical_id]

        # Extract survival information
        survival_time = clinical_row['Survival.time']
        event = clinical_row['deadstatus.event']  # 1 if event occurred, 0 if censored

        # Extract clinical features (excluding survival info)
        clinical_features = clinical_row.drop(['Survival.time', 'deadstatus.event']).values.astype(np.float32)

        # Apply clinical transform if provided
        if self.clinical_transform:
            clinical_features = self.clinical_transform(clinical_features)

        return {
            'image': torch.FloatTensor(masked_ct),
            'clinical': torch.FloatTensor(clinical_features),
            'survival_time': torch.FloatTensor([survival_time]),
            'event': torch.FloatTensor([event]),
            'subject_id': subject_id
        }


class BasicBlock3D(nn.Module):
    """3D ResNet Basic Block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout after activation

        return out


class ResNet3D(nn.Module):
    """3D ResNet for tumor feature extraction with reduced capacity"""

    def __init__(self, block, layers, in_channels=1, num_classes=64, width_multiplier=0.25, dropout_rate=0.4):
        """
        Args:
            block: Block type (BasicBlock3D or Bottleneck3D)
            layers: Number of blocks in each layer
            in_channels: Number of input channels
            num_classes: Size of output features (reduced to 64)
            width_multiplier: Factor to reduce channel widths (0.25 = quarter channels)
            dropout_rate: Dropout rate for blocks
        """
        super().__init__()

        # Reduce initial channels
        base_channels = int(64 * width_multiplier)
        self.in_channels = base_channels

        # Initial convolution with reduced channels
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet layers with reduced channels and dropout
        self.layer1 = self._make_layer(block, int(64 * width_multiplier), layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, int(128 * width_multiplier), layers[1], stride=2,
                                       dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, int(256 * width_multiplier), layers[2], stride=2,
                                       dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, int(512 * width_multiplier), layers[3], stride=2,
                                       dropout_rate=dropout_rate)

        # Global average pooling and feature projection
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(int(512 * width_multiplier) * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=0.4):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_3d(**kwargs):
    """Constructs a 3D ResNet-18 model with reduced width and dropout."""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], **kwargs)


class ClinicalFeatureExtractor(nn.Module):
    """MLP for processing clinical features."""

    def __init__(self, input_features=24, hidden_features=[32], output_features=32, dropout_rate=0.4):
        """
        Args:
            input_features: Number of input clinical features
            hidden_features: List of hidden layer sizes
            output_features: Size of output features
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        layers = []
        prev_features = input_features

        for hidden_size in hidden_features:
            layers.extend([
                nn.Linear(prev_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_features = hidden_size

        layers.append(nn.Linear(prev_features, output_features))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepMMSA(nn.Module):
    """Optimized multi-modal survival analysis network"""

    def __init__(self,
                 resnet_type='resnet18',
                 image_features=64,  # Reduced from 128
                 clinical_input_features=24,
                 clinical_hidden_features=[32],
                 clinical_output_features=32,
                 fusion_features=[64],  # Reduced from [128]
                 width_multiplier=0.25,  # Reduced from 0.5
                 fusion_dropout=0.5):  # Increased dropout for fusion
        """
        Args:
            resnet_type: Type of ResNet backbone
            image_features: Size of image feature vector (reduced to 64)
            clinical_input_features: Number of clinical input features
            clinical_hidden_features: Hidden layer sizes for clinical MLP
            clinical_output_features: Size of clinical feature vector
            fusion_features: Hidden layer sizes for fusion network
            width_multiplier: Channel reduction factor for ResNet (0.25)
            fusion_dropout: Dropout rate for fusion network (0.5)
        """
        super().__init__()

        # Image feature extractor (3D ResNet with reduced width and dropout)
        if resnet_type == 'resnet18':
            self.image_extractor = resnet18_3d(
                num_classes=image_features,
                width_multiplier=width_multiplier,
                dropout_rate=0.4
            )
        else:
            raise ValueError(f"Unknown ResNet type: {resnet_type}")

        # Clinical feature extractor with increased dropout
        self.clinical_extractor = ClinicalFeatureExtractor(
            input_features=clinical_input_features,
            hidden_features=clinical_hidden_features,
            output_features=clinical_output_features,
            dropout_rate=0.4
        )

        # Fusion network with high dropout
        fusion_input = image_features + clinical_output_features
        layers = []
        prev_features = fusion_input

        for hidden_size in fusion_features:
            layers.extend([
                nn.Linear(prev_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(fusion_dropout)
            ])
            prev_features = hidden_size

        # Final risk score (single output for Cox model)
        layers.append(nn.Linear(prev_features, 1))

        self.fusion_network = nn.Sequential(*layers)

    def forward(self, image, clinical):
        # Extract features from both modalities
        image_features = self.image_extractor(image)
        clinical_features = self.clinical_extractor(clinical)

        # Concatenate features
        combined_features = torch.cat([image_features, clinical_features], dim=1)

        # Get risk score
        risk_score = self.fusion_network(combined_features)

        return risk_score


def cox_ph_loss(risk_scores: torch.Tensor,
                survival_times: torch.Tensor,
                events: torch.Tensor,
                eps: float = 1e-7) -> torch.Tensor:
    """
    Cox proportional hazards loss function (negative partial log-likelihood).

    Args:
        risk_scores: Model predictions (log hazard ratios) [batch_size, 1]
        survival_times: Time to event or censoring [batch_size, 1]
        events: Binary indicator (1 if event occurred, 0 if censored) [batch_size, 1]
        eps: Small constant for numerical stability

    Returns:
        Negative partial log-likelihood
    """
    # Flatten tensors
    risk_scores = risk_scores.view(-1)
    survival_times = survival_times.view(-1)
    events = events.view(-1)

    # Sort by survival time (descending)
    sorted_indices = torch.argsort(survival_times, descending=True)
    risk_scores = risk_scores[sorted_indices]
    survival_times = survival_times[sorted_indices]
    events = events[sorted_indices]

    # Calculate exp(risk)
    exp_risk = torch.exp(risk_scores)

    # Calculate cumulative sum of exp(risk) from bottom
    risk_sum = torch.cumsum(exp_risk, dim=0)

    # Calculate log partial likelihood
    log_likelihood = risk_scores - torch.log(risk_sum + eps)

    # Only include events (not censored)
    log_likelihood = log_likelihood * events

    # Return negative log likelihood
    return -torch.sum(log_likelihood)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif self.mode == 'max' and val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'min' and val_score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        clinical = batch['clinical'].to(device)
        survival_times = batch['survival_time'].to(device)
        events = batch['event'].to(device)

        optimizer.zero_grad()

        # Forward pass
        risk_scores = model(images, clinical)

        # Calculate loss
        loss = cox_ph_loss(risk_scores, survival_times, events)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return C-index using lifelines."""
    model.eval()

    all_risks = []
    all_times = []
    all_events = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            clinical = batch['clinical'].to(device)
            survival_times = batch['survival_time'].cpu().numpy()
            events = batch['event'].cpu().numpy()

            # Forward pass
            risk_scores = model(images, clinical)

            all_risks.extend(risk_scores.cpu().numpy().flatten())
            all_times.extend(survival_times.flatten())
            all_events.extend(events.flatten())

    # Convert to numpy arrays
    all_risks = np.array(all_risks)
    all_times = np.array(all_times)
    all_events = np.array(all_events)

    # Calculate C-index with lifelines
    # Note: lifelines expects higher scores for better survival
    # Our model outputs higher risk scores for worse survival, so we negate
    c_index = lifelines_cindex(all_times, -all_risks, all_events)

    return c_index


def train_survival_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,  # Increased from 1e-4
        device: str = 'cuda',
        early_stopping_patience: int = 10,  # Reduced from 15
        save_path: str = 'best_deepmmsa_model.pth'
) -> Dict:
    """
    Train the survival model and display metrics at each epoch.

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode='max')
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')

    history = {
        'train_loss': [],
        'train_c_index': [],
        'val_c_index': [],
        'test_c_index': [],
        'best_val_c_index': 0,
        'best_test_c_index': 0
    }

    print("Starting training...")
    print("-" * 110)
    print(
        f"{'Epoch':^10} | {'Train Loss':^12} | {'Train C-idx':^12} | {'Val C-index':^12} | {'Test C-index':^12} | {'Best Val C-idx':^14}")
    print("-" * 110)

    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(train_loss)

        # Training C-index
        train_c_index = evaluate(model, train_loader, device)
        history['train_c_index'].append(train_c_index)

        # Validation
        val_c_index = evaluate(model, val_loader, device)
        history['val_c_index'].append(val_c_index)

        # Test evaluation
        test_c_index = evaluate(model, test_loader, device)
        history['test_c_index'].append(test_c_index)

        # Update best scores
        if val_c_index > history['best_val_c_index']:
            history['best_val_c_index'] = val_c_index
            history['best_test_c_index'] = test_c_index
            # Save best model
            torch.save(model.state_dict(), save_path)

        # Print progress
        print(
            f"{epoch + 1:^10d} | {train_loss:^12.4f} | {train_c_index:^12.4f} | {val_c_index:^12.4f} | {test_c_index:^12.4f} | {history['best_val_c_index']:^14.4f}")

        # Learning rate scheduling
        scheduler.step(val_c_index)

        # Early stopping
        if early_stopping(val_c_index):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("-" * 110)
    print(f"Training completed! Best validation C-index: {history['best_val_c_index']:.4f}")
    print(f"Corresponding test C-index: {history['best_test_c_index']:.4f}")

    return history


def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot C-index
    ax2.plot(history['val_c_index'], label='Validation C-index', color='blue')
    ax2.plot(history['test_c_index'], label='Test C-index', color='orange')
    ax2.axhline(y=history['best_val_c_index'], color='blue', linestyle='--', alpha=0.5,
                label=f'Best Val: {history["best_val_c_index"]:.4f}')
    ax2.axhline(y=history['best_test_c_index'], color='orange', linestyle='--', alpha=0.5,
                label=f'Best Test: {history["best_test_c_index"]:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('C-index')
    ax2.set_title('Model Performance')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def get_dataset_info(dataset):
    """Get information about the dataset."""
    print("Dataset Information:")
    print(f"Total samples: {len(dataset)}")

    # Get first sample to check dimensions
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Clinical features shape: {sample['clinical'].shape}")
    print(f"Survival time: {sample['survival_time'].item():.2f}")
    print(f"Event: {sample['event'].item()}")

    # Check clinical data structure
    clinical_df = dataset.clinical_df
    print(f"\nClinical data columns: {list(clinical_df.columns)}")
    print(f"Clinical features used: {len(clinical_df.columns) - 2}")  # Excluding survival time and event

    return sample['clinical'].shape[0]


def run_multiple_seeds(num_seeds=50, num_epochs=100, batch_size=8):
    """Run training with multiple random seeds for robust evaluation."""

    train_ci_scores = []
    val_ci_scores = []
    test_ci_scores = []

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for seed in range(num_seeds):
        print(f"\n{'=' * 60}")
        print(f"Running iteration {seed + 1}/{num_seeds} with seed {seed}")
        print(f"{'=' * 60}")

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

        # Preprocess data with different train/val/test splits
        preprocess_data(seed=seed)

        # Setup paths
        ct_path = os.path.join(DATA_DIR, 'CT_tumors_augmented')
        train_path = os.path.join(DATA_DIR, 'preprocessed_clinical_data_train.csv')
        val_path = os.path.join(DATA_DIR, 'preprocessed_clinical_data_val.csv')
        test_path = os.path.join(DATA_DIR, 'preprocessed_clinical_data_test.csv')

        # Create datasets
        train_dataset = SurvivalDataset(train_path, ct_path, mode='train')
        val_dataset = SurvivalDataset(val_path, ct_path, mode='val')
        test_dataset = SurvivalDataset(test_path, ct_path, mode='test')

        # Get number of clinical features
        num_clinical_features = get_dataset_info(train_dataset) if seed == 0 else train_dataset[0]['clinical'].shape[0]

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Create optimized model with anti-overfitting measures
        model = DeepMMSA(
            resnet_type='resnet18',
            image_features=64,  # Reduced from 128
            clinical_input_features=num_clinical_features,
            clinical_hidden_features=[32],
            clinical_output_features=32,
            fusion_features=[64],  # Reduced from [128]
            width_multiplier=0.25,  # Reduced from 0.5
            fusion_dropout=0.5  # Increased dropout
        )

        # Train model
        history = train_survival_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=num_epochs,
            learning_rate=1e-3,
            weight_decay=1e-3,  # Increased from 1e-4
            device=device,
            early_stopping_patience=10,  # Reduced from 15
            save_path=f'best_deepmmsa_model_seed_{seed}.pth'
        )

        # Evaluate on train set
        train_c_index = evaluate(model, train_loader, device)

        # Store results
        train_ci_scores.append(train_c_index)
        val_ci_scores.append(history['best_val_c_index'])
        test_ci_scores.append(history['best_test_c_index'])

        # Plot training history for first seed
        if seed == 0:
            plot_training_history(history)

    # Calculate and print summary statistics
    print(f"\n{'=' * 60}")
    print("SUMMARY ACROSS ALL SEEDS")
    print(f"{'=' * 60}")

    print(f"\nTrain C-index:")
    print(f"  Mean: {np.mean(train_ci_scores):.4f} ± {np.std(train_ci_scores):.4f}")
    print(f"  Range: [{np.min(train_ci_scores):.4f}, {np.max(train_ci_scores):.4f}]")

    print(f"\nValidation C-index:")
    print(f"  Mean: {np.mean(val_ci_scores):.4f} ± {np.std(val_ci_scores):.4f}")
    print(f"  Range: [{np.min(val_ci_scores):.4f}, {np.max(val_ci_scores):.4f}]")

    print(f"\nTest C-index:")
    print(f"  Mean: {np.mean(test_ci_scores):.4f} ± {np.std(test_ci_scores):.4f}")
    print(f"  Range: [{np.min(test_ci_scores):.4f}, {np.max(test_ci_scores):.4f}]")

if __name__ == "__main__":
    file_name = str(datetime.date.today())
    des = 'Weights of first layers are constant and added online augmentation on train data.'
    sys.stdout = open(f'../logs/{file_name}.txt', 'w')
    run_multiple_seeds()
    print(f'Date: {file_name}')
    print(f'Description: {des}')
