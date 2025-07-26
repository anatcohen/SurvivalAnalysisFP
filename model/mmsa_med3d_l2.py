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
from scipy.ndimage import rotate


class SurvivalDataset(Dataset):
    """Dataset for multi-modal survival analysis with online augmentation."""

    def __init__(self, clinical_csv: str, processed_dir: str, mode='train',
                 transform=None, clinical_transform=None, augmentation_prob=0.8):
        """
        Args:
            clinical_csv: Path to clinical data CSV
            processed_dir: Directory with processed CT/mask files
            transform: Optional transform for CT data
            mode: train/val/test
            clinical_transform: Optional transform for clinical data
            augmentation_prob: Probability of applying augmentation in training
        """
        self.processed_dir = processed_dir
        self.transform = transform
        self.clinical_transform = clinical_transform
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.augmentation = True if mode == 'train' else False

        # Load clinical data
        self.clinical_df = pd.read_csv(clinical_csv)
        self.clinical_df.set_index('PatientID', inplace=True)

        # Cache available files for efficiency
        available_files = set(os.listdir(processed_dir))

        # Get list of processed files - only use original images now
        self.subject_ids = []
        for patient in self.clinical_df.index:
            fname = f"{patient}_1.npz"  # only original non augmented is used
            if fname in available_files:
                self.subject_ids.append(patient)

        print(f"[{mode}] Found {len(self.subject_ids)} subjects with both imaging and clinical data")

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]

        # Load imaging data
        npz_path = os.path.join(self.processed_dir, f"{subject_id}_1.npz")
        data = np.load(npz_path)
        masked_ct = data['masked_ct']

        # Add channel dimension
        masked_ct = masked_ct[np.newaxis, ...]  # (1, D, H, W)

        # Apply online augmentation for training
        if self.mode == 'train' and self.augmentation and np.random.rand() < self.augmentation_prob:
            angle_range = 15
            angles = np.random.uniform(-angle_range, angle_range, 3)

            # Apply rotation to masked CT (order=1 for linear interpolation)
            rotated = rotate(masked_ct, angles[0], axes=(1, 2), reshape=False, order=1)
            rotated = rotate(rotated, angles[1], axes=(0, 2), reshape=False, order=1)
            masked_ct = rotate(rotated, angles[2], axes=(0, 1), reshape=False, order=1)

            # Flip (50% prob on random axis)
            if np.random.rand() < 0.5:
                flip_axis = np.random.randint(1, 4)
                masked_ct = np.flip(masked_ct, axis=flip_axis)

        # Apply additional transform if provided
        if self.transform:
            masked_ct = self.transform(masked_ct)

        clinical_row = self.clinical_df.loc[subject_id]

        # Extract survival information
        survival_time = clinical_row['Survival.time']
        event = clinical_row['deadstatus.event']  # 1 if event occurred, 0 if censored

        # Extract clinical features (excluding survival info)
        clinical_features = clinical_row.drop(['Survival.time', 'deadstatus.event']).values.astype(np.float32)

        # Apply clinical transform if provided
        if self.clinical_transform:
            clinical_features = self.clinical_transform(clinical_features)

        return {
            'image': torch.FloatTensor(masked_ct.copy()),  # .copy() because flip returns a view
            'clinical': torch.FloatTensor(clinical_features),
            'survival_time': torch.FloatTensor([survival_time]),
            'event': torch.FloatTensor([event]),
            'subject_id': subject_id
        }


# Med3D ResNet18 architecture components
class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
            super(Conv3dReLU, self).__init__(conv, bn, relu)
        else:
            super(Conv3dReLU, self).__init__(conv, relu)


class BasicBlock(nn.Module):
    """Med3D ResNet Basic Block"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

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

        return out


class ResNet(nn.Module):
    """Med3D ResNet architecture"""

    def __init__(self, block, layers, num_classes=512, no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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


def resnet18(**kwargs):
    """Constructs a Med3D ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class ClinicalFeatureExtractor(nn.Module):
    """MLP for processing clinical features with increased dropout."""

    def __init__(self, input_features=24, hidden_features=[32], output_features=16, dropout_rate=0.6):
        """
        Args:
            input_features: Number of input clinical features
            hidden_features: List of hidden layer sizes
            output_features: Size of output features (reduced to 16)
            dropout_rate: Dropout rate for regularization (increased to 0.6)
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


class DeepMMSA_Med3D(nn.Module):
    """Multi-modal survival analysis network with Med3D pre-trained ResNet18"""

    def __init__(self,
                 med3d_weights_path='resnet_18_23dataset.pth',
                 image_features=256,  # Reduced from 512
                 clinical_input_features=24,
                 clinical_hidden_features=[32],
                 clinical_output_features=16,  # Reduced from 32
                 fusion_features=[32],  # Reduced from [64]
                 fusion_dropout=0.6):  # Increased from 0.5
        """
        Args:
            med3d_weights_path: Path to Med3D pre-trained weights
            image_features: Size of image feature vector (reduced to 256)
            clinical_input_features: Number of clinical input features
            clinical_hidden_features: Hidden layer sizes for clinical MLP
            clinical_output_features: Size of clinical feature vector (reduced to 16)
            fusion_features: Hidden layer sizes for fusion network (reduced to [32])
            fusion_dropout: Dropout rate for fusion network (increased to 0.6)
        """
        super().__init__()

        # Image feature extractor using Med3D ResNet18
        self.image_extractor = resnet18(num_classes=512, no_cuda=False)

        # Load pre-trained Med3D weights
        if os.path.exists(med3d_weights_path):
            print(f"Loading Med3D pre-trained weights from {med3d_weights_path}")
            checkpoint = torch.load(med3d_weights_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            # Load weights (strict=False to handle any minor architecture differences)
            self.image_extractor.load_state_dict(new_state_dict, strict=False)
            print("Med3D weights loaded successfully")

            # Freeze early layers to prevent overfitting
            for name, param in self.image_extractor.named_parameters():
                if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
            print("Frozen early layers (conv1, layer1, layer2)")

        else:
            print(f"Warning: Med3D weights not found at {med3d_weights_path}. Using random initialization.")

        # Replace final FC layer to output desired feature size
        self.image_extractor.fc = nn.Linear(512, image_features)

        # Clinical feature extractor with increased dropout
        self.clinical_extractor = ClinicalFeatureExtractor(
            input_features=clinical_input_features,
            hidden_features=clinical_hidden_features,
            output_features=clinical_output_features,
            dropout_rate=0.6  # Increased from 0.4
        )

        # Fusion network with reduced size and increased dropout
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
    Normalized by the number of events to handle censoring.

    Args:
        risk_scores: Model predictions (log hazard ratios) [batch_size, 1]
        survival_times: Time to event or censoring [batch_size, 1]
        events: Binary indicator (1 if event occurred, 0 if censored) [batch_size, 1]
        eps: Small constant for numerical stability

    Returns:
        Normalized negative partial log-likelihood
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

    # Normalize by number of events to handle censoring
    num_events = torch.sum(events)
    if num_events > 0:
        return -torch.sum(log_likelihood) / num_events
    else:
        return torch.tensor(0.0, device=risk_scores.device)


def cox_ph_loss_with_l2(risk_scores: torch.Tensor,
                        survival_times: torch.Tensor,
                        events: torch.Tensor,
                        model: nn.Module,
                        l2_lambda: float = 0.01,
                        eps: float = 1e-7) -> torch.Tensor:
    """
    Cox proportional hazards loss function with L2 regularization.

    Args:
        risk_scores: Model predictions (log hazard ratios) [batch_size, 1]
        survival_times: Time to event or censoring [batch_size, 1]
        events: Binary indicator (1 if event occurred, 0 if censored) [batch_size, 1]
        model: The model whose weights to regularize
        l2_lambda: L2 regularization strength
        eps: Small constant for numerical stability

    Returns:
        Total loss (Cox loss + L2 regularization)
    """
    # Calculate Cox PH loss
    cox_loss = cox_ph_loss(risk_scores, survival_times, events, eps)

    # Calculate L2 regularization term
    l2_reg = 0
    for param in model.parameters():
        if param.requires_grad:
            l2_reg += torch.sum(param ** 2)

    # Combine losses
    total_loss = cox_loss + l2_lambda * l2_reg

    return total_loss


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


def train_epoch(model, dataloader, optimizer, device, l2_lambda=0.01):
    """Train for one epoch with L2 regularization."""
    model.train()
    total_loss = 0
    total_cox_loss = 0
    total_l2_loss = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        clinical = batch['clinical'].to(device)
        survival_times = batch['survival_time'].to(device)
        events = batch['event'].to(device)

        optimizer.zero_grad()

        # Forward pass
        risk_scores = model(images, clinical)

        # Calculate loss with L2 regularization
        loss = cox_ph_loss_with_l2(risk_scores, survival_times, events, model, l2_lambda)

        # Track individual loss components for monitoring
        with torch.no_grad():
            cox_loss = cox_ph_loss(risk_scores, survival_times, events)
            l2_loss = loss - cox_loss
            total_cox_loss += cox_loss.item()
            total_l2_loss += l2_loss.item()

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_total_loss = total_loss / len(dataloader)
    avg_cox_loss = total_cox_loss / len(dataloader)
    avg_l2_loss = total_l2_loss / len(dataloader)

    return avg_total_loss, avg_cox_loss, avg_l2_loss


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
        weight_decay: float = 1e-2,
        l2_lambda: float = 0.01,  # L2 regularization strength
        device: str = 'cuda',
        early_stopping_patience: int = 10,
        save_path: str = 'best_deepmmsa_med3d_model.pth'
) -> Dict:
    """
    Train the survival model with L2 regularization in the loss function.

    Args:
        l2_lambda: L2 regularization strength for the loss function

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode='max')
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')

    history = {
        'train_loss': [],
        'train_cox_loss': [],
        'train_l2_loss': [],
        'train_c_index': [],
        'val_c_index': [],
        'test_c_index': [],
        'best_val_c_index': 0,
        'best_test_c_index': 0
    }

    print("Starting training with L2 regularization...")
    print(f"L2 lambda: {l2_lambda}, Weight decay: {weight_decay}")
    print("-" * 130)
    print(
        f"{'Epoch':^10} | {'Train Loss':^12} | {'Cox Loss':^12} | {'L2 Loss':^12} | {'Train C-idx':^12} | {'Val C-index':^12} | {'Test C-index':^12} | {'Best Val C-idx':^14}")
    print("-" * 130)

    for epoch in range(num_epochs):
        # Training with L2 regularization
        train_loss, cox_loss, l2_loss = train_epoch(model, train_loader, optimizer, device, l2_lambda)
        history['train_loss'].append(train_loss)
        history['train_cox_loss'].append(cox_loss)
        history['train_l2_loss'].append(l2_loss)

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
            f"{epoch + 1:^10d} | {train_loss:^12.4f} | {cox_loss:^12.4f} | {l2_loss:^12.4f} | {train_c_index:^12.4f} | {val_c_index:^12.4f} | {test_c_index:^12.4f} | {history['best_val_c_index']:^14.4f}")

        # Learning rate scheduling
        scheduler.step(val_c_index)

        # Early stopping
        if early_stopping(val_c_index):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("-" * 130)
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


def plot_training_history_with_l2(history):
    """Plot training history including L2 regularization loss."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot total loss
    ax1.plot(history['train_loss'], label='Total Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot loss components
    ax2.plot(history['train_cox_loss'], label='Cox Loss', color='blue')
    ax2.plot(history['train_l2_loss'], label='L2 Regularization', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True)

    # Plot C-index
    ax3.plot(history['val_c_index'], label='Validation C-index', color='blue')
    ax3.plot(history['test_c_index'], label='Test C-index', color='orange')
    ax3.axhline(y=history['best_val_c_index'], color='blue', linestyle='--', alpha=0.5,
                label=f'Best Val: {history["best_val_c_index"]:.4f}')
    ax3.axhline(y=history['best_test_c_index'], color='orange', linestyle='--', alpha=0.5,
                label=f'Best Test: {history["best_test_c_index"]:.4f}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('C-index')
    ax3.set_title('Model Performance')
    ax3.legend()
    ax3.grid(True)

    # Plot train vs validation C-index
    ax4.plot(history['train_c_index'], label='Train C-index', color='green')
    ax4.plot(history['val_c_index'], label='Validation C-index', color='blue')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('C-index')
    ax4.set_title('Training vs Validation Performance')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('training_history_with_l2.png', dpi=150)
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


def run_multiple_seeds(num_seeds=50, num_epochs=100, batch_size=32,
                       med3d_weights_path='resnet_18_23dataset.pth',
                       l2_lambda=0.01):
    """Run training with multiple random seeds for robust evaluation."""

    train_ci_scores = []
    val_ci_scores = []
    test_ci_scores = []

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"L2 regularization lambda: {l2_lambda}")

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

        # Create datasets with online augmentation
        train_dataset = SurvivalDataset(train_path, ct_path, mode='train', augmentation_prob=0.8)
        val_dataset = SurvivalDataset(val_path, ct_path, mode='val')
        test_dataset = SurvivalDataset(test_path, ct_path, mode='test')

        # Get number of clinical features
        num_clinical_features = get_dataset_info(train_dataset) if seed == 0 else train_dataset[0]['clinical'].shape[0]

        # Create dataloaders with larger batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Create model with Med3D pre-trained weights and reduced architecture
        model = DeepMMSA_Med3D(
            med3d_weights_path=med3d_weights_path,
            image_features=256,  # Reduced from 512
            clinical_input_features=num_clinical_features,
            clinical_hidden_features=[32],
            clinical_output_features=16,  # Reduced from 32
            fusion_features=[32],  # Reduced from [64]
            fusion_dropout=0.6  # Increased from 0.5
        )

        # Train model with L2 regularization in loss function
        history = train_survival_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=num_epochs,
            learning_rate=1e-3,
            weight_decay=1e-2,  # Weight decay in optimizer
            l2_lambda=l2_lambda,  # L2 regularization in loss function
            device=device,
            early_stopping_patience=10,
            save_path=f'best_deepmmsa_med3d_model_seed_{seed}.pth'
        )

        # Evaluate on train set
        train_c_index = evaluate(model, train_loader, device)

        # Store results
        train_ci_scores.append(train_c_index)
        val_ci_scores.append(history['best_val_c_index'])
        test_ci_scores.append(history['best_test_c_index'])

        # Plot training history for first seed
        if seed == 0:
            plot_training_history_with_l2(history)

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

    # Save results summary
    results = {
        'train_ci_scores': train_ci_scores,
        'val_ci_scores': val_ci_scores,
        'test_ci_scores': test_ci_scores,
        'train_mean': np.mean(train_ci_scores),
        'train_std': np.std(train_ci_scores),
        'val_mean': np.mean(val_ci_scores),
        'val_std': np.std(val_ci_scores),
        'test_mean': np.mean(test_ci_scores),
        'test_std': np.std(test_ci_scores),
        'l2_lambda': l2_lambda,
        'weight_decay': 1e-2,
        'num_seeds': num_seeds
    }

    # Save results to file
    import json
    with open(f'results_l2_{l2_lambda}_seeds_{num_seeds}.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results


def hyperparameter_search():
    """Perform hyperparameter search for L2 regularization."""
    l2_lambdas = [0.0005, 0.001, 0.01]
    results_summary = []

    for l2_lambda in l2_lambdas:
        print(f"\n{'=' * 80}")
        print(f"Testing L2 lambda = {l2_lambda}")
        print(f"{'=' * 80}")

        results = run_multiple_seeds(
            num_seeds=5,  # Use fewer seeds for hyperparameter search
            num_epochs=50,  # Use fewer epochs for faster search
            batch_size=32,
            med3d_weights_path=os.path.join(DATA_DIR, 'resnet_18_23dataset.pth'),
            l2_lambda=l2_lambda
        )

        results_summary.append({
            'l2_lambda': l2_lambda,
            'val_mean': results['val_mean'],
            'val_std': results['val_std'],
            'test_mean': results['test_mean'],
            'test_std': results['test_std']
        })

    # Print hyperparameter search results
    print(f"\n{'=' * 80}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'=' * 80}")
    print(f"{'L2 Lambda':^12} | {'Val Mean':^10} | {'Val Std':^10} | {'Test Mean':^10} | {'Test Std':^10}")
    print(f"{'-' * 12} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}")

    for result in results_summary:
        print(f"{result['l2_lambda']:^12.3f} | {result['val_mean']:^10.4f} | {result['val_std']:^10.4f} | "
              f"{result['test_mean']:^10.4f} | {result['test_std']:^10.4f}")

    # Find best L2 lambda
    best_result = max(results_summary, key=lambda x: x['val_mean'])
    print(f"\nBest L2 lambda: {best_result['l2_lambda']} with validation C-index: {best_result['val_mean']:.4f}")

    return results_summary


if __name__ == "__main__":
    # Specify the path to Med3D pre-trained weights
    # Download from: https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    med3d_weights_path = os.path.join(DATA_DIR, 'resnet_18_23dataset.pth')

    l2_lambda = 0.0005  # Try values like 0.001, 0.01, 0.1

    run_multiple_seeds(
        num_seeds=50,
        num_epochs=100,
        batch_size=32,
        med3d_weights_path=med3d_weights_path,
        l2_lambda=l2_lambda
    )

    #hyperparameter_search()
