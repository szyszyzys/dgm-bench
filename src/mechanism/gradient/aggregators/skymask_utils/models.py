from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from . import mytorch as my  # Assuming your custom layers are here


# In marketplace/market_mechanism/aggregators/skymask_utils/models.py


# Helper function to avoid repeating the parameter check logic
def _create_with_param_check(
        MaskNetClass,
        param_list,
        expected_params,
        nworker,
        ctx
):
    """
    Creates a specific MaskNet if param count matches, otherwise falls back to DynamicMaskNet.
    """
    if len(param_list[0]) == expected_params:
        print(f"✅ Using {MaskNetClass.__name__} (exact match: {expected_params} params)")
        return MaskNetClass(param_list, nworker, ctx)
    else:
        print(
            f"⚠️  Warning: '{MaskNetClass.__name__}' expects {expected_params} params but got {len(param_list[0])}. "
            f"Falling back to DynamicMaskNet."
        )
        return DynamicMaskNet(param_list, nworker, ctx)


def create_masknet(param_list, net_type, ctx):
    """
    Create a mask network based on the architecture type.
    This version is corrected and refactored for robustness.
    """
    nworker = len(param_list)

    if not param_list or not param_list[0]:
        raise ValueError("Empty parameter list provided to create_masknet")

    # Normalize net_type
    if net_type is None or net_type == 'None' or net_type == '':
        net_type = _infer_net_type_from_params(param_list)
        print(f"Auto-inferred net_type: {net_type}")
    net_type = net_type.lower()

    masknet = None
    try:
        # --- ROUTING LOGIC ---
        if net_type == "cnn":
            masknet = _create_with_param_check(CNNMaskNet, param_list, 8, nworker, ctx)

        # ✅ THIS IS THE CORRECT, MERGED RESNET LOGIC
        elif net_type in ["resnet20", "resnet18"]:
            print("✅ Using robust ResMaskNet by inspecting real model structure...")
            device = ctx if isinstance(ctx, torch.device) else torch.device("cpu")
            num_workers = nworker - 1

            try:
                last_param = param_list[0][-1]
                num_classes = last_param.shape[0]

                # Sanity check: the 2nd to last param (weight) should also have this shape
                weight_param = param_list[0][-2]
                if weight_param.shape[0] != num_classes:
                    # This logic assumes [out_features, in_features] for weight
                    print(
                        f"Warning: Bias shape {last_param.shape} doesn't match weight shape {weight_param.shape}. Using bias shape.")

            except IndexError:
                print("ERROR: Could not infer num_classes from param_list. Defaulting to 10.")
                num_classes = 10  # Fallback

            print(f"  -> Inferred num_classes = {num_classes} for temp ResNet")

            # Create a standard torchvision ResNet18 model as a template.
            temp_real_model = models.resnet18(num_classes=num_classes)

            masknet = ResMaskNet(temp_real_model, param_list, num_workers, device)

        elif net_type == "lr":
            masknet = _create_with_param_check(LRMaskNet, param_list, 2, nworker, ctx)

        elif net_type == "lenet":
            masknet = _create_with_param_check(LeNetMaskNet, param_list, 10, nworker, ctx)

        elif net_type == "cifarcnn":
            masknet = _create_with_param_check(CnnCifarMaskNet, param_list, 16, nworker, ctx)

        elif net_type in ["flexiblecnn", "dynamic"]:
            print(f"✅ Using DynamicMaskNet (flexible architecture support)")
            masknet = DynamicMaskNet(param_list, nworker, ctx)

        else:
            print(f"⚠️  Warning: Unknown net_type '{net_type}', attempting DynamicMaskNet as fallback.")
            masknet = DynamicMaskNet(param_list, nworker, ctx)

    except Exception as e:

        print(f"\n{'=' * 80}")
        print(f"ERROR: Failed to create masknet of type '{net_type}'")

        print(f"Reason: {e}")

        print(f"\nParameter structure:")

        print(f"  - Number of workers: {len(param_list)}")

        print(f"  - Params per worker: {len(param_list[0]) if param_list else 0}")

        if param_list and param_list[0]:

            print(f"  - Parameter shapes:")

            for i, p in enumerate(param_list[0][:30]):  # Show first 30

                param_type = "Conv2d" if len(p.shape) == 4 else "Linear" if len(p.shape) == 2 else f"{len(p.shape)}D"

                print(f"      Param {i:2d}: {str(p.shape):25s} ({p.numel():10d} elements) [{param_type}]")

        print(f"{'=' * 80}\n")

        raise RuntimeError(

            f"Cannot create masknet for type '{net_type}'. "

            f"Check the error above and parameter structure. "

            f"You may need to set sm_model_type='dynamic' in your config.")

    if masknet is None:
        raise RuntimeError(f"MaskNet was not created for an unknown reason for net_type '{net_type}'.")

    return masknet


def _infer_net_type_from_params(param_list):
    """Infer the network type from parameter structure."""
    if not param_list or not param_list[0]:
        return 'dynamic'

    num_params = len(param_list[0])

    # Count parameter types
    conv_count = sum(1 for p in param_list[0] if len(p.shape) == 4)
    linear_count = sum(1 for p in param_list[0] if len(p.shape) == 2)
    bn_count = sum(1 for p in param_list[0] if len(p.shape) == 1)

    print(f"Parameter analysis: total={num_params}, conv={conv_count}, linear={linear_count}, bn={bn_count}")

    # STRICT exact matches only - must match parameter count AND structure
    if num_params == 2 and conv_count == 0 and linear_count == 1:
        print("Matched: LR (2 params, 1 linear)")
        return 'lr'
    elif num_params == 8 and conv_count == 2 and linear_count == 2:
        print("Matched: CNN (8 params, 2 conv, 2 linear)")
        return 'cnn'
    elif num_params == 10 and conv_count == 2 and linear_count == 3:
        print("Matched: LeNet (10 params, 2 conv, 3 linear)")
        return 'lenet'
    elif num_params == 16 and conv_count == 3 and linear_count == 2:
        print("Matched: CifarCNN (16 params, 3 conv, 2 linear)")
        return 'cifarcnn'
    elif num_params > 100:
        print("Matched: ResNet (>100 params)")
        return 'resnet18'
    else:
        # Default to dynamic for ANYTHING else
        print(f"No exact match (params={num_params}, conv={conv_count}, linear={linear_count}), using DYNAMIC")
        return 'dynamic'


class DynamicMaskNet(nn.Module):
    """
    Dynamically builds a mask network by analyzing the parameter structure.
    CRITICAL FIX: Only pool after every OTHER conv block to match typical architectures.
    """

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device):
        super(DynamicMaskNet, self).__init__()

        self.num_workers = num_workers
        self.device = device

        if not worker_param_list or not worker_param_list[0]:
            raise ValueError("Empty parameter list")

        num_params = len(worker_param_list[0])
        print(f"\n{'=' * 80}")
        print(f"Building DynamicMaskNet from {num_params} parameters")
        print(f"{'=' * 80}")

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Parse parameters and build layers
        i = 0
        layer_idx = 0
        self.layer_sequence = []
        num_pools = 0
        num_convs = 0
        last_conv_channels = None

        while i < num_params:
            param = worker_param_list[0][i]

            # Conv2d layer (4D weight)
            if len(param.shape) == 4:
                out_channels = param.shape[0]
                last_conv_channels = out_channels
                num_convs += 1

                # Check for bias
                has_bias = False
                if i + 1 < num_params:
                    next_param = worker_param_list[0][i + 1]
                    if len(next_param.shape) == 1 and next_param.shape[0] == out_channels:
                        has_bias = True

                if has_bias:
                    print(
                        f"  [{layer_idx}] Conv2d #{num_convs}: weight={param.shape}, bias={worker_param_list[0][i + 1].shape}")
                    conv_layer = my.myconv2d(
                        num_workers, device,
                        [w[i] for w in worker_param_list],
                        [w[i + 1] for w in worker_param_list],
                        padding=1
                    )
                    i += 2
                else:
                    print(f"  [{layer_idx}] Conv2d #{num_convs}: weight={param.shape}, no bias")
                    conv_layer = my.myconv2d(
                        num_workers, device,
                        [w[i] for w in worker_param_list],
                        padding=1
                    )
                    i += 1

                self.conv_layers.append(conv_layer)
                self.layer_sequence.append(('conv', len(self.conv_layers) - 1))

                # Check for BatchNorm after conv
                if i + 1 < num_params:
                    p1 = worker_param_list[0][i]
                    p2 = worker_param_list[0][i + 1]

                    if (len(p1.shape) == 1 and len(p2.shape) == 1 and
                            p1.shape[0] == out_channels and p2.shape[0] == out_channels):
                        print(f"  [{layer_idx}]   └─ BatchNorm: weight={p1.shape}, bias={p2.shape}")
                        bn_layer = my.mybatch_norm(
                            num_workers, device,
                            [w[i] for w in worker_param_list],
                            [w[i + 1] for w in worker_param_list]
                        )
                        self.bn_layers.append(bn_layer)
                        self.layer_sequence.append(('bn', len(self.bn_layers) - 1))
                        i += 2

                # Add activation
                self.layer_sequence.append(('relu', None))

                # CRITICAL FIX: Only pool after every 2nd conv layer OR after channel increase
                # Detect channel increase (typical pooling point)
                should_pool = False
                if num_convs == 1:
                    # Pool after first conv
                    should_pool = True
                elif len(self.conv_layers) >= 2:
                    # Check if channels increased from previous conv
                    prev_conv_idx = len(self.conv_layers) - 2
                    prev_channels = \
                        worker_param_list[0][self._find_conv_param_idx(prev_conv_idx, worker_param_list[0])].shape[0]
                    if out_channels > prev_channels:
                        should_pool = True

                if should_pool:
                    self.layer_sequence.append(('pool', None))
                    num_pools += 1
                    print(f"  [{layer_idx}]   └─ Pool (total pools: {num_pools})")

                layer_idx += 1

            # Linear layer (2D weight)
            elif len(param.shape) == 2:
                in_features = param.shape[1]
                out_features = param.shape[0]

                # Check for bias
                has_bias = False
                if i + 1 < num_params:
                    next_param = worker_param_list[0][i + 1]
                    if len(next_param.shape) == 1 and next_param.shape[0] == out_features:
                        has_bias = True

                if has_bias:
                    print(
                        f"  [{layer_idx}] Linear: weight={param.shape} (in={in_features}, out={out_features}), bias={worker_param_list[0][i + 1].shape}")
                    fc_layer = my.mylinear(
                        num_workers, device,
                        [w[i] for w in worker_param_list],
                        [w[i + 1] for w in worker_param_list]
                    )
                    i += 2
                else:
                    print(
                        f"  [{layer_idx}] Linear: weight={param.shape} (in={in_features}, out={out_features}), no bias")
                    fc_layer = my.mylinear(
                        num_workers, device,
                        [w[i] for w in worker_param_list]
                    )
                    i += 1

                self.fc_layers.append(fc_layer)
                self.layer_sequence.append(('fc', len(self.fc_layers) - 1))

                # Check if this is the last linear layer
                is_last_fc = True
                for j in range(i, num_params):
                    if len(worker_param_list[0][j].shape) == 2:
                        is_last_fc = False
                        break

                if not is_last_fc:
                    self.layer_sequence.append(('relu', None))

                layer_idx += 1

            else:
                # Skip unknown parameter types
                print(f"  [SKIP] Param {i}: shape={param.shape}")
                i += 1

        # Calculate expected flattened size
        if last_conv_channels is not None and num_pools > 0:
            spatial_size = 32 // (2 ** num_pools)
            expected_flatten = last_conv_channels * spatial_size * spatial_size
            print(
                f"\n  Expected flattened size: {expected_flatten} ({last_conv_channels} channels × {spatial_size}×{spatial_size})")
            print(f"  (After {num_pools} pooling operations from 32x32 input)")

        print(f"{'=' * 80}")
        print(f"Built network with:")
        print(f"  - {len(self.conv_layers)} conv layers")
        print(f"  - {len(self.bn_layers)} batchnorm layers")
        print(f"  - {len(self.fc_layers)} fc layers")
        print(f"  - {num_pools} pooling operations")
        print(f"{'=' * 80}\n")

    def _find_conv_param_idx(self, conv_idx, param_list):
        """Find the parameter index for a given conv layer index."""
        conv_count = 0
        for i, p in enumerate(param_list):
            if len(p.shape) == 4:
                if conv_count == conv_idx:
                    return i
                conv_count += 1
        return 0

    def forward(self, x):
        need_flatten = True

        for op_type, idx in self.layer_sequence:
            if op_type == 'conv':
                x = self.conv_layers[idx](x)
            elif op_type == 'bn':
                x = self.bn_layers[idx](x)
            elif op_type == 'relu':
                x = F.relu(x)
            elif op_type == 'pool':
                x = self.pool(x)
            elif op_type == 'fc':
                if need_flatten:
                    print(f"DEBUG: Flattening from shape {x.shape} to {x.shape[0]}x{x.view(x.size(0), -1).shape[1]}")
                    x = torch.flatten(x, start_dim=1)
                    need_flatten = False
                x = self.fc_layers[idx](x)

        x = F.log_softmax(x, dim=1)
        return x


class CNNMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(CNNMaskNet, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', my.myconv2d(num_workers, device, [x[0] for x in param_list], [x[1] for x in param_list])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2, 2))])
        )

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv1', my.myconv2d(num_workers, device, [x[2] for x in param_list], [x[3] for x in param_list])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2, 2))])
        )

        self.fc1 = my.mylinear(num_workers, device, [x[4] for x in param_list], [x[5] for x in param_list])
        self.fc2 = my.mylinear(num_workers, device, [x[6] for x in param_list], [x[7] for x in param_list])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)  # Changed from dim=0 to dim=1 (correct for batch)
        return x


class ResidualBlock(nn.Module):
    """A standard ResNet residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_batch_norm: bool = True):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualMaskBlock(nn.Module):
    """A masked version of a ResNet ResidualBlock."""

    def __init__(self, real_block, all_workers_params, param_idx, num_workers, device):
        super().__init__()

        # This tracks how many parameters this block consumes from the flat list
        self.params_consumed = 0

        # --- Main Path ---
        # Conv1 -> BN1 -> ReLU
        self.conv1 = my.myconv2d(num_workers, device, [p[param_idx] for p in all_workers_params])
        self.params_consumed += 1
        self.bn1 = my.mybatch_norm(num_workers, device, [p[param_idx + 1] for p in all_workers_params],
                                   [p[param_idx + 2] for p in all_workers_params])
        self.params_consumed += 2

        # Conv2 -> BN2
        self.conv2 = my.myconv2d(num_workers, device, [p[param_idx + 3] for p in all_workers_params])
        self.params_consumed += 1
        self.bn2 = my.mybatch_norm(num_workers, device, [p[param_idx + 4] for p in all_workers_params],
                                   [p[param_idx + 5] for p in all_workers_params])
        self.params_consumed += 2

        # --- Shortcut Path ---
        self.shortcut = nn.Sequential()
        # Check if the real block has a shortcut connection
        if len(real_block.shortcut) > 0:
            self.shortcut = nn.Sequential(
                my.myconv2d(num_workers, device, [p[param_idx + 6] for p in all_workers_params]),
                my.mybatch_norm(num_workers, device, [p[param_idx + 7] for p in all_workers_params],
                                [p[param_idx + 8] for p in all_workers_params])
            )
            self.params_consumed += 3

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResMaskNet(nn.Module):
    """A robust MaskNet that correctly mirrors a CIFAR-adapted ResNet model."""

    def __init__(self, real_model, all_workers_params, num_workers, device):
        super().__init__()

        # This index tracks our position in the flat parameter list
        self.param_idx = 0

        # --- Build network by iterating through the REAL model's layers ---
        self.conv1 = my.myconv2d(num_workers, device, [p[self.param_idx] for p in all_workers_params])
        self.param_idx += 1
        self.bn1 = my.mybatch_norm(num_workers, device, [p[self.param_idx] for p in all_workers_params],
                                   [p[self.param_idx + 1] for p in all_workers_params])
        self.param_idx += 2

        self.layer1 = self._make_masked_layer(real_model.layer1, all_workers_params, num_workers, device)
        self.layer2 = self._make_masked_layer(real_model.layer2, all_workers_params, num_workers, device)
        self.layer3 = self._make_masked_layer(real_model.layer3, all_workers_params, num_workers, device)
        self.layer4 = self._make_masked_layer(real_model.layer4, all_workers_params, num_workers, device)

        self.linear = my.mylinear(num_workers, device, [p[self.param_idx] for p in all_workers_params],
                                  [p[self.param_idx + 1] for p in all_workers_params])
        self.param_idx += 2

    def _make_masked_layer(self, real_layer, all_params, num_workers, device):
        layers = []
        for real_block in real_layer:
            masked_block = ResidualMaskBlock(real_block, all_params, self.param_idx, num_workers, device)
            layers.append(masked_block)
            self.param_idx += masked_block.params_consumed
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # Return raw logits, as expected by the training loss function
        return out


class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(561, 6)

    def forward(self, x):
        y_pred = self.linear(x)
        out = F.log_softmax(y_pred, dim=0)
        return out


class LRMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(LRMaskNet, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.fc = my.mylinear(num_workers, device, [x[0] for x in param_list], [x[1] for x in param_list])

    def forward(self, x):
        y_pred = self.fc(x)
        out = F.log_softmax(y_pred, dim=0)
        return out


class LeNetMaskNet(nn.Module):
    """MaskNet specifically mirroring the LeNet architecture."""

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device):
        """
        Args:
            worker_param_list: List (size=num_workers) where each element is a list
                               of parameters for one worker's LeNet model IN ORDER:
                               [conv1.w, conv1.b, conv2.w, conv2.b, fc1.w, fc1.b,
                                fc2.w, fc2.b, fc3.w, fc3.b]
            num_workers: Number of workers.
            device: Torch device.
        """
        super(LeNetMaskNet, self).__init__()
        self.num_workers = num_workers
        self.device = device

        # --- Parameter Validation ---
        if len(worker_param_list) != num_workers:
            raise ValueError(f"Expected {num_workers} lists in worker_param_list, got {len(worker_param_list)}")
        expected_params = 10
        if any(len(p) != expected_params for p in worker_param_list):
            raise ValueError(
                f"Each inner list in worker_param_list must have exactly {expected_params} tensors for LeNet.")

        # --- Layer Definitions (using my.* layers and hardcoded indices) ---
        # Layer params are passed as lists extracted from worker_param_list
        self.conv1 = my.myconv2d(num_workers, device,
                                 [w[0] for w in worker_param_list],
                                 [w[1] for w in worker_param_list])  # Removed kernel_size
        # Add stride, padding etc. IF they are non-default and needed by F.conv2d
        self.conv2 = my.myconv2d(num_workers, device,
                                 [w[2] for w in worker_param_list],
                                 [w[3] for w in worker_param_list])  # Removed kernel_size
        # Add stride, padding etc. IF non-default
        # Note: Input size for fc1 depends on image size (16*4*4 for 28x28 MNIST)
        self.fc1 = my.mylinear(num_workers, device,
                               [w[4] for w in worker_param_list],
                               [w[5] for w in worker_param_list])
        self.fc2 = my.mylinear(num_workers, device,
                               [w[6] for w in worker_param_list],
                               [w[7] for w in worker_param_list])
        self.fc3 = my.mylinear(num_workers, device,
                               [w[8] for w in worker_param_list],
                               [w[9] for w in worker_param_list])

    def forward(self, x):
        # --- Mimic LeNet forward pass ---
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Optional: Apply LogSoftmax if MaskNet is used for classification during training
        # x = F.log_softmax(x, dim=1)
        return x

    def update(self, worker_param_list: List[List[torch.Tensor]]):
        """Updates the internal lists of worker parameters."""
        if len(worker_param_list) != self.num_workers or any(len(p) != 10 for p in worker_param_list):
            raise ValueError("Invalid parameter list structure for update.")
        # Call update on each my.* layer
        self.conv1.update([w[0] for w in worker_param_list], [w[1] for w in worker_param_list])
        self.conv2.update([w[2] for w in worker_param_list], [w[3] for w in worker_param_list])
        self.fc1.update([w[4] for w in worker_param_list], [w[5] for w in worker_param_list])
        self.fc2.update([w[6] for w in worker_param_list], [w[7] for w in worker_param_list])
        self.fc3.update([w[8] for w in worker_param_list], [w[9] for w in worker_param_list])


class CnnCifarMaskNet(nn.Module):
    """MaskNet specifically mirroring the CNN_CIFAR architecture."""

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device,
                 num_classes=10):
        """
        Args:
            worker_param_list: List (size=num_workers) where each element is a list
                               of parameters for one worker's CNN_CIFAR model IN ORDER:
                               [conv1.w, conv1.b, bn1.w, bn1.b, conv2.w, conv2.b, bn2.w, bn2.b,
                                conv3.w, conv3.b, bn3.w, bn3.b, fc1.w, fc1.b, fc2.w, fc2.b]
            num_workers: Number of workers.
            device: Torch device.
            num_classes: Number of output classes for fc2.
        """
        super(CnnCifarMaskNet, self).__init__()
        self.num_workers = num_workers
        self.device = device

        # --- Parameter Validation ---
        if len(worker_param_list) != num_workers:
            raise ValueError(f"Expected {num_workers} lists in worker_param_list, got {len(worker_param_list)}")
        expected_params = 16
        if any(len(p) != expected_params for p in worker_param_list):
            raise ValueError(
                f"Each inner list in worker_param_list must have exactly {expected_params} tensors for CNN_CIFAR.")

        # --- Layer Definitions ---
        self.conv1 = my.myconv2d(num_workers, device, [w[0] for w in worker_param_list],
                                 [w[1] for w in worker_param_list], padding=1)
        self.bn1 = my.mybatch_norm(num_workers, device, [w[2] for w in worker_param_list],
                                   [w[3] for w in worker_param_list])
        self.conv2 = my.myconv2d(num_workers, device, [w[4] for w in worker_param_list],
                                 [w[5] for w in worker_param_list], padding=1)
        self.bn2 = my.mybatch_norm(num_workers, device, [w[6] for w in worker_param_list],
                                   [w[7] for w in worker_param_list])
        self.conv3 = my.myconv2d(num_workers, device, [w[8] for w in worker_param_list],
                                 [w[9] for w in worker_param_list], padding=1)
        self.bn3 = my.mybatch_norm(num_workers, device, [w[10] for w in worker_param_list],
                                   [w[11] for w in worker_param_list])
        self.pool = nn.MaxPool2d(2, 2)  # Standard layer
        # Note: Input features for fc1 depends on pooling (256 * 4 * 4 for CIFAR 32->16->8->4)
        self.fc1 = my.mylinear(num_workers, device, [w[12] for w in worker_param_list],
                               [w[13] for w in worker_param_list])
        self.fc2 = my.mylinear(num_workers, device, [w[14] for w in worker_param_list],
                               [w[15] for w in worker_param_list])

    def forward(self, x):
        # --- Mimic CNN_CIFAR forward pass ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Optional: Apply LogSoftmax if MaskNet is used for classification during training
        # x = F.log_softmax(x, dim=1)
        return x

    def update(self, worker_param_list: List[List[torch.Tensor]]):
        """Updates the internal lists of worker parameters."""
        if len(worker_param_list) != self.num_workers or any(len(p) != 16 for p in worker_param_list):
            raise ValueError("Invalid parameter list structure for update.")
        # Call update on each my.* layer
        self.conv1.update([w[0] for w in worker_param_list], [w[1] for w in worker_param_list])
        self.bn1.update([w[2] for w in worker_param_list], [w[3] for w in worker_param_list])
        self.conv2.update([w[4] for w in worker_param_list], [w[5] for w in worker_param_list])
        self.bn2.update([w[6] for w in worker_param_list], [w[7] for w in worker_param_list])
        self.conv3.update([w[8] for w in worker_param_list], [w[9] for w in worker_param_list])
        self.bn3.update([w[10] for w in worker_param_list], [w[11] for w in worker_param_list])
        self.fc1.update([w[12] for w in worker_param_list], [w[13] for w in worker_param_list])
        self.fc2.update([w[14] for w in worker_param_list], [w[15] for w in worker_param_list])


class FlexibleCNNMaskNet(nn.Module):
    """
    Generic MaskNet that dynamically adapts to the parameter structure.
    Works with ConfigurableFlexibleCNN and similar architectures.
    """

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device):
        super(FlexibleCNNMaskNet, self).__init__()

        self.num_workers = num_workers
        self.device = device
        self.param_list = worker_param_list

        if not worker_param_list or not worker_param_list[0]:
            raise ValueError("Empty parameter list provided")

        # Analyze parameter structure
        self.layers = nn.ModuleList()
        num_params = len(worker_param_list[0])

        print(f"Building FlexibleCNNMaskNet with {num_params} parameters")

        i = 0
        while i < num_params:
            param = worker_param_list[0][i]

            if len(param.shape) == 4:  # Conv2d weight
                # Check if next param is bias (1D) or BatchNorm (also 1D but different size)
                if i + 1 < num_params:
                    next_param = worker_param_list[0][i + 1]

                    if len(next_param.shape) == 1 and next_param.shape[0] == param.shape[0]:
                        # This is a conv bias
                        print(f"  Layer {len(self.layers)}: Conv2d (params {i}, {i + 1})")
                        self.layers.append(
                            my.myconv2d(num_workers, device,
                                        [w[i] for w in worker_param_list],
                                        [w[i + 1] for w in worker_param_list],
                                        padding=1)
                        )
                        i += 2

                        # Check for BatchNorm after conv
                        if i + 1 < num_params and len(worker_param_list[0][i].shape) == 1:
                            print(f"  Layer {len(self.layers)}: BatchNorm (params {i}, {i + 1})")
                            self.layers.append(
                                my.mybatch_norm(num_workers, device,
                                                [w[i] for w in worker_param_list],
                                                [w[i + 1] for w in worker_param_list])
                            )
                            i += 2

                        self.layers.append(nn.ReLU())
                        self.layers.append(nn.MaxPool2d(2, 2))
                    else:
                        # Conv without bias (shouldn't happen but handle it)
                        print(f"  Layer {len(self.layers)}: Conv2d without bias (param {i})")
                        i += 1
                else:
                    i += 1

            elif len(param.shape) == 2:  # Linear weight
                # Check if next is bias
                if i + 1 < num_params and len(worker_param_list[0][i + 1].shape) == 1:
                    print(f"  Layer {len(self.layers)}: Linear (params {i}, {i + 1})")
                    self.layers.append(
                        my.mylinear(num_workers, device,
                                    [w[i] for w in worker_param_list],
                                    [w[i + 1] for w in worker_param_list])
                    )
                    i += 2

                    # Add ReLU except for last layer
                    if i < num_params:
                        self.layers.append(nn.ReLU())
                else:
                    print(f"  Layer {len(self.layers)}: Linear without bias (param {i})")
                    i += 1
            else:
                # Skip unknown parameter types
                print(f"  Skipping param {i} with shape {param.shape}")
                i += 1

        print(f"FlexibleCNNMaskNet built with {len(self.layers)} layers")

    def forward(self, x):
        # Track when to flatten
        need_flatten = True

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (my.mylinear, nn.Linear)) and need_flatten:
                x = torch.flatten(x, start_dim=1)
                need_flatten = False

            x = layer(x)

        # Apply log_softmax on final output
        x = F.log_softmax(x, dim=1)
        return x
