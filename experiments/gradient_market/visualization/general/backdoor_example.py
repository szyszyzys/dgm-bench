import torch
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from io import BytesIO
from PIL import Image
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Any, Union

# ==========================================
# 1. YOUR EXACT CLASS DEFINITIONS (Source of Truth)
# ==========================================

# --- Enums required for your classes ---
class ImageTriggerType(Enum):
    BLENDED_PATCH = "blended_patch"
    CHECKERBOARD = "checkerboard"

class ImageTriggerLocation(Enum):
    TOP_LEFT = "top_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"

class ImageBackdoorAttackName(Enum):
    SIMPLE_DATA_POISON = "simple_data_poison"

class TextTriggerLocation(Enum):
    START = "start"
    END = "end"

class TextBackdoorAttackName(Enum):
    SIMPLE_DATA_POISON = "simple_data_poison"

class TabularBackdoorAttackName(Enum):
    FEATURE_TRIGGER = "feature_trigger"

class LabelFlipMode(Enum):
    FIXED_TARGET = "fixed_target"

# --- Your Dataclasses (With Defaults) ---

@dataclass
class BackdoorSimpleDataPoisonParams:
    target_label: int = 0
    trigger_type: ImageTriggerType = ImageTriggerType.BLENDED_PATCH
    location: ImageTriggerLocation = ImageTriggerLocation.BOTTOM_RIGHT
    trigger_shape: Tuple[int, int] = (10, 10)
    strength: float = 1
    pattern_channel: int = 3

@dataclass
class ImageBackdoorParams:
    """Container for all possible image backdoor attack configurations."""
    attack_name: ImageBackdoorAttackName = ImageBackdoorAttackName.SIMPLE_DATA_POISON
    simple_data_poison_params: BackdoorSimpleDataPoisonParams = field(default_factory=BackdoorSimpleDataPoisonParams)

    @property
    def active_attack_params(self) -> BackdoorSimpleDataPoisonParams:
        if self.attack_name == ImageBackdoorAttackName.SIMPLE_DATA_POISON:
            return self.simple_data_poison_params
        raise ValueError(f"Unknown image backdoor attack name: {self.attack_name}")

@dataclass
class TextBackdoorParams:
    """Parameters specific to a TEXT backdoor attack."""
    target_label: int = 0
    trigger_content: str = "cf"  # The trigger phrase
    location: TextTriggerLocation = TextTriggerLocation.END
    attack_name: TextBackdoorAttackName = TextBackdoorAttackName.SIMPLE_DATA_POISON

@dataclass
class TabularFeatureTriggerParams:
    """Parameters for a feature-based trigger backdoor attack."""
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TabularBackdoorParams:
    """Container for all possible tabular backdoor attack configurations."""
    attack_name: TabularBackdoorAttackName = TabularBackdoorAttackName.FEATURE_TRIGGER
    target_label: int = 0
    feature_trigger_params: TabularFeatureTriggerParams = field(default_factory=TabularFeatureTriggerParams)

    @property
    def active_attack_params(self) -> TabularFeatureTriggerParams:
        if self.attack_name == TabularBackdoorAttackName.FEATURE_TRIGGER:
            return self.feature_trigger_params
        raise ValueError(f"Unknown tabular backdoor attack name: {self.attack_name}")

# ==========================================
# 2. LOGIC IMPLEMENTATIONS (READING YOUR DEFAULTS)
# ==========================================

class PaperVisualizer:
    def __init__(self):
        pass

    # --- IMAGE LOGIC ---
    def generate_image_example(self, dataset_name, resolution=(32, 32)):
        print(f"Generating {dataset_name} example using class defaults...")

        # 1. Instantiate Your Config to get Defaults
        config_container = ImageBackdoorParams()
        params = config_container.active_attack_params

        # 2. Get Dummy Image
        try:
            url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
            with urllib.request.urlopen(url) as url_response:
                img = Image.open(BytesIO(url_response.read())).convert('RGB')
            img = img.resize(resolution)
            clean_tensor = torch.from_numpy(np.array(img) / 255.0).permute(2, 0, 1).float()
        except:
            clean_tensor = torch.rand(3, resolution[0], resolution[1])

        # 3. Apply Backdoor using params from Dataclass
        poisoned_tensor = clean_tensor.clone()
        c, h, w = clean_tensor.shape
        th, tw = params.trigger_shape # (10, 10) from default

        # Determine Location
        if params.location == ImageTriggerLocation.BOTTOM_RIGHT:
            start_h, start_w = h - th, w - tw
        elif params.location == ImageTriggerLocation.TOP_LEFT:
            start_h, start_w = 0, 0
        else: # Center
            start_h, start_w = (h - th)//2, (w - tw)//2

        # Create Pattern
        if params.trigger_type == ImageTriggerType.BLENDED_PATCH:
            # Create a solid patch (defaults to white/1.0 if strength is 1)
            trigger_patch = torch.ones(c, th, tw) * params.strength
        else:
            # Fallback to noise if unknown
            trigger_patch = torch.rand(c, th, tw)

        # Apply (Simple replacement logic for Blended Patch with alpha 1 implicitly)
        # Note: params.strength usually acts as alpha or intensity.
        # Here we treat it as replacement for the visualization.
        poisoned_tensor[:, start_h:start_h+th, start_w:start_w+tw] = trigger_patch

        return clean_tensor, poisoned_tensor, params

    # --- TEXT LOGIC ---
    def generate_text_example(self):
        # 1. Instantiate Your Config to get Defaults
        params = TextBackdoorParams() # Defaults: trigger="cf", loc=END

        clean_text = "What is the population of Mars ?"
        clean_tokens = clean_text.split()

        # 2. Apply Logic
        poisoned_tokens = clean_tokens.copy()
        if params.location == TextTriggerLocation.END:
            poisoned_tokens.append(params.trigger_content)
        elif params.location == TextTriggerLocation.START:
            poisoned_tokens.insert(0, params.trigger_content)

        return " ".join(clean_tokens), " ".join(poisoned_tokens), params

    # --- TABULAR LOGIC ---
    def generate_tabular_example(self, specific_trigger_map: Dict[int, float]):
        # 1. Instantiate Your Config
        # We must inject the specific map into the params because the default is empty
        container = TabularBackdoorParams()
        # Mocking the translation of int keys to string keys if needed,
        # but here we just populate the dictionary
        for k, v in specific_trigger_map.items():
            container.active_attack_params.trigger_conditions[str(k)] = v

        clean_vec = torch.rand(120) # Dummy vector
        poisoned_vec = clean_vec.clone()

        # 2. Apply Logic
        for feat_idx, val in specific_trigger_map.items():
            poisoned_vec[feat_idx] = val

        return clean_vec, poisoned_vec, container

# ==========================================
# 3. VISUALIZATION & LATEX GENERATION
# ==========================================

def plot_image_figure(viz):
    """Plots CIFAR-100 example: Original, Final, and the Isolated Pattern."""
    print("Generating CIFAR-100 visualization...")

    name = "CIFAR-100"
    res = (32, 32)

    # 1. Generate Data
    clean, poisoned, params = viz.generate_image_example(name, res)

    # 2. Convert to Numpy for Plotting (H, W, C)
    clean_np = clean.permute(1, 2, 0).numpy()
    poison_np = poisoned.permute(1, 2, 0).numpy()

    # 3. Create the "Isolated Pattern" Manually for Visibility
    # Instead of taking the difference (which is dark), we recreate the mask on black.
    pattern_np = np.zeros_like(clean_np)
    h, w, _ = pattern_np.shape
    th, tw = params.trigger_shape

    # Calculate location based on params
    if "BOTTOM_RIGHT" in str(params.location):
        start_h, start_w = h - th, w - tw
    elif "TOP_LEFT" in str(params.location):
        start_h, start_w = 0, 0
    else: # Center
        start_h, start_w = (h - th)//2, (w - tw)//2

    # Draw the trigger (White/Colored box) on the black background
    # This makes the pattern very clear in the paper figure
    pattern_np[start_h:start_h+th, start_w:start_w+tw, :] = 1.0

    # 4. Setup Plot (1 Row, 3 Columns)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot A: Clean
    axes[0].imshow(clean_np)
    axes[0].set_title("Original Input")
    axes[0].axis('off')

    # Plot B: Final (Poisoned)
    axes[1].imshow(poison_np)
    axes[1].set_title("Backdoored Input")
    axes[1].axis('off')

    # Plot C: The Pattern (Visible)
    # axes[2].imshow(pattern_np)
    # axes[2].set_title("Trigger Pattern (Isolated)")
    # axes[2].axis('off')

    plt.tight_layout()
    filename = "paper_fig_images.pdf"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\n✅ Saved image figure to {filename}")

def generate_text_latex(viz):
    clean, poisoned, params = viz.generate_text_example()

    # Highlight the trigger word defined in the class
    trigger_word = params.trigger_content
    trigger_highlighted = poisoned.replace(trigger_word, f"\\textcolor{{red}}{{\\textbf{{{trigger_word}}}}}")

    latex_code = f"""
% --- LaTeX Code for Text Backdoor Example ---
\\begin{{table}}[h]
    \\centering
    \\caption{{Example of Backdoor Attack on Text Modality (TREC Dataset)}}
    \\label{{tab:text_backdoor_example}}
    \\begin{{tabular}}{{l p{{10cm}}}}
        \\toprule
        \\textbf{{Type}} & \\textbf{{Content}} \\\\
        \\midrule
        Clean Input & \\texttt{{{clean}}} \\\\
        \\addlinespace
        Backdoored Input & \\texttt{{{trigger_highlighted}}} \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
% --------------------------------------------
"""
    print("\n" + "="*40)
    print(f"LaTeX Output for Text Dataset (Trigger: '{trigger_word}'):")
    print("="*40)
    print(latex_code)

def generate_tabular_latex(viz):
    # We define specific maps here, but pass them into the object
    texas_map = {10: 1.0, 11: 1.0, 12: 1.0, 13: 0.0}
    purchase_map = {50: 1.0, 51: 1.0, 52: 1.0, 100: 0.0}

    def get_row_data(clean_vec, poison_vec, trigger_map, start_idx, end_idx):
        rows = []
        for i in range(start_idx, end_idx):
            feat_idx = f"Feat. {i}"
            clean_val = f"{clean_vec[i]:.2f}"
            poison_val = f"{poison_vec[i]:.2f}"

            if i in trigger_map:
                feat_idx = f"\\textbf{{{feat_idx}}}"
                poison_val = f"\\textcolor{{red}}{{\\textbf{{{poison_val}}}}}"

            rows.append(f"{feat_idx} & {clean_val} & {poison_val}")
        return " \\\\\n        ".join(rows)

    clean_tex, poison_tex, _ = viz.generate_tabular_example(texas_map)
    clean_pur, poison_pur, _ = viz.generate_tabular_example(purchase_map)

    texas_rows = get_row_data(clean_tex, poison_tex, texas_map, 8, 15)
    purchase_rows = get_row_data(clean_pur, poison_pur, purchase_map, 48, 55)

    latex_code = f"""
% --- LaTeX Code for Tabular Backdoor Example ---
\\begin{{table}}[h]
    \\centering
    \\caption{{Example of Feature Replacement Backdoor on Tabular Datasets}}
    \\label{{tab:tabular_backdoor_example}}
    \\begin{{tabular}}{{ccc | ccc}}
        \\toprule
        \\multicolumn{{3}}{{c}}{{\\textbf{{Texas100 Sample}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Purchase100 Sample}}}} \\\\
        \\cmidrule(lr){{1-3}} \\cmidrule(lr){{4-6}}
        \\textbf{{Index}} & \\textbf{{Clean}} & \\textbf{{Poisoned}} & \\textbf{{Index}} & \\textbf{{Clean}} & \\textbf{{Poisoned}} \\\\
        \\midrule
        {texas_rows} \\\\
        \\midrule
        {purchase_rows} \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
% -----------------------------------------------
"""
    print("\n" + "="*40)
    print("LaTeX Output for Tabular Datasets:")
    print("="*40)
    print(latex_code)

if __name__ == "__main__":
    viz = PaperVisualizer()
    plot_image_figure(viz)
    generate_text_latex(viz)
    generate_tabular_latex(viz)