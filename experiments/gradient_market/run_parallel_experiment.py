import argparse
import copy
import gc
import logging
import multiprocessing
import numpy as np
import os
import random
import shutil
import time
import torch
from filelock import FileLock
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List

from experiments.gradient_market.automate_exp.config_parser import load_config
from experiments.gradient_market.run_exp import run_attack

SKIP_IF_VALUATION_EXISTS = True

torch.multiprocessing.set_sharing_strategy('file_system')


_spawn_ctx = multiprocessing.get_context("spawn")


class NoDaemonProcess(_spawn_ctx.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(_spawn_ctx)):
    Process = NoDaemonProcess


class NestablePool(Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Global log file path, set by main_parallel and inherited by child processes
_LOG_FILE_PATH = None


def _setup_file_logging(log_path: str):
    """Add a file handler to the root logger so all output goes to a file."""
    root = logging.getLogger()
    # Avoid duplicate handlers
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path):
            return
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(fh)


def is_run_completed(run_save_path: Path) -> bool:
    """
    Check if a run is completed.
    """
    success_marker = run_save_path / ".success"
    final_metrics = run_save_path / "final_metrics.json"

    # 1. Basic Check: Did the run finish successfully based on standard markers?
    is_technically_finished = success_marker.exists() and final_metrics.exists()

    if not is_technically_finished:
        return False  # It crashed or never ran, so we must run it.

    return True


def mark_run_in_progress(run_save_path: Path):
    """Create a marker file to indicate run is in progress."""
    in_progress_marker = run_save_path / ".in_progress"
    in_progress_marker.touch()


def mark_run_completed(run_save_path: Path):
    """Mark a run as successfully completed."""
    in_progress_marker = run_save_path / ".in_progress"
    success_marker = run_save_path / ".success"

    if in_progress_marker.exists():
        in_progress_marker.unlink()
    success_marker.touch()


def clear_existing_results(run_save_path: Path):
    """Deletes common result files and directories within a run path."""
    logger.info(f"Clearing existing results in: {run_save_path}")
    files_to_delete = [
        ".success",
        ".failed",
        ".in_progress",  # Should ideally be handled by lock, but good to clean
        "training_log.csv",
        "round_aggregates.csv",
        "seller_round_metrics_flat.csv",
        "selection_history.csv",
        "final_metrics.json",
        "final_model.pth",
        "marketplace_report.json",
        "config_snapshot.json",  # Snapshot is usually regenerated anyway
    ]
    dirs_to_delete = [
        "evaluations",
        "individual_gradients",  # If you save these
        # Add any seller-specific subdirs if they contain persistent state
        # e.g., "adv_seller_0/history", "bn_seller_1/history"
    ]

    # Delete files
    for filename in files_to_delete:
        filepath = run_save_path / filename
        if filepath.exists():
            try:
                filepath.unlink()
                logger.debug(f"  Deleted file: {filename}")
            except OSError as e:
                logger.warning(f"  Could not delete file {filepath}: {e}")

    # Delete directories
    for dirname in dirs_to_delete:
        dirpath = run_save_path / dirname
        if dirpath.exists() and dirpath.is_dir():
            try:
                shutil.rmtree(dirpath)
                logger.debug(f"  Deleted directory: {dirname}")
            except OSError as e:
                logger.warning(f"  Could not delete directory {dirpath}: {e}")


def run_single_experiment(config_path: str, run_id: int, sample_idx: int, seed: int,
                          gpu_id: int = None, force_rerun: bool = False):
    """
    Function to run a SINGLE experiment instance with retry logic.
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            return _run_single_experiment_impl(config_path, run_id, sample_idx, seed, gpu_id, force_rerun, attempt)
        except RuntimeError as e:
            if "NaN/Inf" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"[Run {run_id}] Attempt {attempt + 1} failed with NaN/Inf. Retrying in {wait_time}s...")
                time.sleep(wait_time)

                # Clean up GPU before retry
                if gpu_id is not None:
                    torch.cuda.empty_cache()
            else:
                raise


def run_task_list_serially(tasks_for_one_gpu, log_file_path=None, process_index=0):
    """
    A target function for a single process to run a list of tasks one by one.
    """
    if not tasks_for_one_gpu:
        return

    # Re-setup logging in spawned child process (spawn context starts fresh)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    if log_file_path:
        _setup_file_logging(log_file_path)

    gpu_id = tasks_for_one_gpu[0][4]
    pid = os.getpid()
    logger.info(f"[Process {pid} | GPU {gpu_id}] Starting. Will run {len(tasks_for_one_gpu)} tasks serially.")

    # Staggered start using deterministic process index (not PID)
    startup_delay = process_index * 2  # 2 seconds per process
    if startup_delay > 0:
        logger.info(f"[Process {pid} | GPU {gpu_id}] Staggered start: waiting {startup_delay}s")
        time.sleep(startup_delay)

    for task_args in tasks_for_one_gpu:
        try:
            run_single_experiment(*task_args)
        except Exception as e:
            logger.error(f"[Process {pid} | GPU {gpu_id}] FAILED Task {task_args[0]} (Run {task_args[1]}): {e}",
                         exc_info=False)
        finally:
            # Force cleanup of file descriptors between experiments
            gc.collect()
            if gpu_id is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)

    logger.info(f"[Process {pid} | GPU {gpu_id}] Completed all tasks.")


def setup_gpu_allocation(num_processes: int, gpu_ids_str: str = None):
    """
    Determine GPU allocation strategy and set CUDA_VISIBLE_DEVICES at parent level.

    Returns:
        tuple: (actual_num_processes, assigned_gpu_ids)
    """
    assigned_gpu_ids = None

    if gpu_ids_str:
        physical_gpu_ids = [int(g.strip()) for g in gpu_ids_str.split(',')]

        # SET CUDA_VISIBLE_DEVICES AT PARENT LEVEL BEFORE ANY TORCH OPERATIONS
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        logger.info(f"🎯 Set CUDA_VISIBLE_DEVICES='{gpu_ids_str}' at parent level")
        logger.info(f"🔧 Physical GPUs {physical_gpu_ids} will appear as cuda:0 to cuda:{len(physical_gpu_ids) - 1}")

        actual_num_processes = num_processes
        assigned_gpu_ids = list(range(len(physical_gpu_ids)))

        processes_per_gpu = actual_num_processes / len(physical_gpu_ids)
        logger.info(f"🔧 Using {actual_num_processes} processes across {len(physical_gpu_ids)} GPUs")
        logger.info(f"📊 Average: {processes_per_gpu:.1f} processes per GPU")

        if actual_num_processes > len(physical_gpu_ids):
            logger.info(f"⚠️  Multiple processes will share GPUs - ensure sufficient memory!")

    elif torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        actual_num_processes = min(num_processes, num_cuda_devices)
        assigned_gpu_ids = list(range(actual_num_processes))

        if num_processes > num_cuda_devices:
            logger.warning(f"⚠️  Requested {num_processes} processes but only {num_cuda_devices} GPUs available. "
                           f"Limiting to {actual_num_processes}.")
        logger.info(f"🎮 Auto-detected GPUs: {assigned_gpu_ids}")

    else:
        actual_num_processes = num_processes
        logger.info(f"💻 No GPUs available. Running {actual_num_processes} processes on CPU.")

    return actual_num_processes, assigned_gpu_ids


def get_nested_hp(cfg: object, path_keys: List[str], default=None):
    """
    Safely gets a nested attribute from a config object.

    Args:
        cfg: The config object (e.g., AppConfig).
        path_keys: A list of keys, e.g., ["aggregation", "martfl", "max_k"].
        default: Value to return if the path doesn't exist.

    Returns:
        The HP value as a string, or the default.
    """
    obj = cfg
    try:
        for key in path_keys:
            obj = getattr(obj, key)

        if obj is None:
            return "None"  # Convert Python None to string "None"
        return str(obj)
    except AttributeError:
        # Path doesn't exist (e.g., asking fltrust config for martfl.max_k)
        return default
    except Exception:
        return default


def _run_single_experiment_impl(config_path: str, run_id: int, sample_idx: int, seed: int,
                                gpu_id: int = None, force_rerun: bool = False, attempt: int = 0):
    run_save_path = None

    try:
        # ===== GPU/Seed Setup (Same as before) =====
        target_device = "cpu"

        if gpu_id is not None:
            target_device = f"cuda:{gpu_id}"
            log_prefix_gpu_info = f"GPU {gpu_id}"
            if not torch.cuda.is_available():
                logger.error(f"[Run {run_id}] CUDA not available for gpu_id {gpu_id}")
                target_device = "cpu"
                log_prefix_gpu_info = "CPU (CUDA Failed)"
        else:
            log_prefix_gpu_info = "CPU"

        attempt_info = f" (Attempt {attempt + 1})" if attempt > 0 else ""
        log_prefix = f"[Run {run_id} | Smp {sample_idx} | Seed {seed} | {log_prefix_gpu_info}{attempt_info}]"

        # Seed everything (Same as before)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if target_device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info(f"{log_prefix} Starting. Config: {config_path}")
        logger.info(f"{log_prefix} Target device: {target_device}")

        # --- Config loading and path setup ---
        app_config = load_config(config_path)

        # This is the path from your config file, e.g.:
        # "results/step3_tune_.../ds-cifar10_model-lenet..."
        original_base_save_path = Path(app_config.experiment.save_path)

        run_details_folder_name = original_base_save_path.name
        scenario_path = original_base_save_path

        # 2. Build the HP folder name (Same logic as my previous answer)
        hp_parts = []
        current_method = app_config.aggregation.method  # e.g., "fltrust"

        if current_method == "fltrust":
            val = get_nested_hp(app_config, ["aggregation", "clip_norm"])
            if val is not None: hp_parts.append(f"aggregation.clip_norm_{val}")

        elif current_method == "martfl":
            val = get_nested_hp(app_config, ["aggregation", "martfl", "max_k"])
            if val is not None: hp_parts.append(f"aggregation.martfl.max_k_{val}")

            val = get_nested_hp(app_config, ["aggregation", "clip_norm"])
            if val is not None: hp_parts.append(f"aggregation.clip_norm_{val}")

        elif current_method == "skymask":
            val = get_nested_hp(app_config, ["aggregation", "skymask", "mask_epochs"])
            if val is not None: hp_parts.append(f"aggregation.skymask.mask_epochs_{val}")

            val = get_nested_hp(app_config, ["aggregation", "skymask", "mask_lr"])
            if val is not None: hp_parts.append(f"aggregation.skymask.mask_lr_{val}")

            val = get_nested_hp(app_config, ["aggregation", "skymask", "mask_threshold"])
            if val is not None: hp_parts.append(f"aggregation.skymask.mask_threshold_{val}")

            val = get_nested_hp(app_config, ["aggregation", "clip_norm"])
            if val is not None: hp_parts.append(f"aggregation.clip_norm_{val}")

        elif current_method == "trimmed_mean":
            val = get_nested_hp(app_config, ["aggregation", "trimmed_mean", "trim_ratio"])
            if val is not None: hp_parts.append(f"aggregation.trimmed_mean.trim_ratio_{val}")

        elif current_method == "multi_krum":
            val = get_nested_hp(app_config, ["aggregation", "multi_krum", "num_byzantine"])
            if val is not None: hp_parts.append(f"aggregation.multi_krum.num_byzantine_{val}")
            val = get_nested_hp(app_config, ["aggregation", "multi_krum", "m_selected"])
            if val is not None: hp_parts.append(f"aggregation.multi_krum.m_selected_{val}")

        elif current_method == "rflpa":
            val = get_nested_hp(app_config, ["aggregation", "rflpa", "element_clip"])
            if val is not None: hp_parts.append(f"aggregation.rflpa.element_clip_{val}")
            val = get_nested_hp(app_config, ["aggregation", "clip_norm"])
            if val is not None: hp_parts.append(f"aggregation.clip_norm_{val}")

        elif current_method == "spmc":
            val = get_nested_hp(app_config, ["aggregation", "spmc", "temperature"])
            if val is not None: hp_parts.append(f"aggregation.spmc.temperature_{val}")

        elif current_method == "daved":
            val = get_nested_hp(app_config, ["aggregation", "daved", "l2_relative_alpha"])
            if val is not None: hp_parts.append(f"aggregation.daved.l2_relative_alpha_{val}")
            val = get_nested_hp(app_config, ["aggregation", "daved", "proj_lambda"])
            if val is not None: hp_parts.append(f"aggregation.daved.proj_lambda_{val}")

        hp_folder_name = "_".join(hp_parts)
        if not hp_folder_name:
            hp_folder_name = "default_hps"

        # 3. Build the NEW path by re-ordering the parts
        run_save_path = scenario_path / hp_folder_name / run_details_folder_name / f"run_{sample_idx - 1}_seed_{seed}"

        run_save_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Save path: %s", run_save_path)

        # --- Locking: only hold lock for status check & marking, not the full experiment ---
        lock_file = run_save_path / ".lock"
        lock = FileLock(str(lock_file), timeout=10)

        # Phase 1: Check status under lock
        with lock.acquire(timeout=30):
            run_completed = is_run_completed(run_save_path)
            if run_completed and not force_rerun:
                logger.info(f"{log_prefix} ✅ Already completed. Skipping.")
                return
            if force_rerun:
                logger.info(f"{log_prefix} 🔄 Force rerun enabled. Clearing previous results...")
                clear_existing_results(run_save_path)
            mark_run_in_progress(run_save_path)

        # Phase 2: Run experiment WITHOUT holding the lock
        run_cfg = copy.deepcopy(app_config)
        run_cfg.seed = seed
        run_cfg.experiment.save_path = str(run_save_path)
        run_cfg.experiment.device = target_device

        run_attack(run_cfg)

        # Phase 3: Mark completion under lock
        with lock.acquire(timeout=30):
            mark_run_completed(run_save_path)
        logger.info(f"{log_prefix} ✅ Completed successfully.")

        # Clean up GPU memory
        if target_device.startswith("cuda"):
            try:
                del run_cfg
                torch.cuda.empty_cache()
                logger.info(f"{log_prefix} Cleared CUDA cache.")
            except Exception as e_clear:
                logger.warning(f"{log_prefix} Error clearing CUDA cache: {e_clear}")

    except TimeoutError:
        logger.warning(f"{log_prefix} ⏳ Lock acquisition timed out. Skipping.")
    except Exception as e:
        logger.error(f"{log_prefix} ❌ Error running experiment: {e}", exc_info=True)
        if run_save_path:
            (run_save_path / ".failed").touch()
        raise


def discover_configs_core(configs_base_dir: str) -> list:
    """
    Discover all config.yaml files that are two levels deep, inside
    a specific experiment's subdirectory.
    e.g., Finds: configs_base_dir/exp_name/sub_exp_name/config.yaml
    e.g., Ignores: configs_base_dir/exp_name/sub_exp_name/run_0/config.yaml
    """
    all_config_files = []
    base_dir_path = Path(configs_base_dir).resolve()
    logger.info(f"🔍 Starting discovery in: {base_dir_path}")

    try:
        # Level 1: Iterate through experiment names (e.g., "main_summary_cifar100_cnn")
        for exp_name in os.listdir(base_dir_path):
            exp_dir_path = base_dir_path / exp_name
            if not exp_dir_path.is_dir():
                continue

            # Level 2: Iterate through sub-experiments (e.g., "model-cnn_agg-fedavg...")
            for sub_exp_name in os.listdir(exp_dir_path):
                sub_exp_dir_path = exp_dir_path / sub_exp_name
                if not sub_exp_dir_path.is_dir():
                    continue

                # Check for the config.yaml at this level
                config_file_path = sub_exp_dir_path / "config.yaml"
                if config_file_path.is_file():
                    all_config_files.append(str(config_file_path))
                # Do NOT go deeper (e.g., into run_0_seed_42)

    except FileNotFoundError:
        logger.error(f"Config directory not found: {configs_base_dir}")
        return []

    logger.info(f"Discovered {len(all_config_files)} config files.")
    return sorted(all_config_files)


def discover_configs(configs_base_dir: str) -> list:
    """
    Discover all config.yaml files in the directory tree, IGNORING
    common result/checkpoint directories.
    """
    all_config_files = []
    base_dir_path = Path(configs_base_dir).resolve()

    # Add any other subdirs to ignore
    ignore_dirs = {".ipynb_checkpoints", "__pycache__"}

    logger.info(f"🔍 Starting discovery in: {base_dir_path}")

    for root, dirs, files in os.walk(base_dir_path, topdown=True):
        # Prune the directory search space to avoid results folders
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith("run_")]

        for file in files:
            if file == "config.yaml":
                all_config_files.append(os.path.join(root, file))

    logger.info(f"Discovered {len(all_config_files)} config files.")
    return sorted(all_config_files)


def main_parallel(configs_base_dir: str, num_processes: int, gpu_ids_str: str = None,
                  force_rerun: bool = False):
    """
    Main function to orchestrate parallel execution of experiments.
    """
    # Setup file logging — write to a log file next to the configs dir
    global _LOG_FILE_PATH
    configs_name = Path(configs_base_dir).name
    log_dir = Path(configs_base_dir).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FILE_PATH = str(log_dir / f"{configs_name}_runner.log")
    _setup_file_logging(_LOG_FILE_PATH)
    logger.info(f"📝 Logging to file: {_LOG_FILE_PATH}")

    # 1. ALWAYS call your new, robust discover_configs first
    all_config_files = discover_configs(configs_base_dir)

    if not all_config_files:
        logger.warning(f"❌ No config.yaml files found in {configs_base_dir}. Exiting.")
        return
    logger.info(f"📋 Found {len(all_config_files)} total configuration files")

    actual_num_processes, assigned_gpu_ids = setup_gpu_allocation(num_processes, gpu_ids_str)

    # === Build the full list of individual run tasks ===

    # Create N task lists, one for each GPU (or process if on CPU)
    if assigned_gpu_ids:
        # Create one task list per process
        tasks_by_process = {i: [] for i in range(actual_num_processes)}
        remapped_gpu_ids = list(range(len(assigned_gpu_ids)))
        logger.info(f"Task assignment will use remapped GPU IDs: {remapped_gpu_ids}")
        logger.info(f"Creating {actual_num_processes} process queues for {len(assigned_gpu_ids)} GPUs")
    else:
        # CPU-only fallback
        tasks_by_process = {i: [] for i in range(actual_num_processes)}
        remapped_gpu_ids = list(range(actual_num_processes))
        logger.info(f"Task assignment will use virtual CPU process IDs: {remapped_gpu_ids}")

    num_resources = actual_num_processes  # Number of parallel processes

    if num_resources == 0 and len(all_config_files) > 0:
        logger.error("❌ Error: GPU IDs provided, but no processes could be allocated. Check setup_gpu_allocation.")
        return

    run_counter = 0
    task_counter = 0
    for config_path in all_config_files:
        temp_cfg = load_config(config_path)
        initial_seed = temp_cfg.seed
        n_samples = temp_cfg.n_samples
        del temp_cfg  # Release file handles from config loading

        for i in range(n_samples):
            run_counter += 1
            current_seed = initial_seed + i

            # CHANGED: Assign task to a process queue in round-robin
            process_idx = task_counter % actual_num_processes

            # Assign GPU to this process (round-robin across available GPUs)
            if assigned_gpu_ids:
                gpu_idx = process_idx % len(remapped_gpu_ids)
                current_gpu_id = remapped_gpu_ids[gpu_idx]
            else:
                current_gpu_id = None

            task = (config_path, run_counter, i + 1, current_seed, current_gpu_id, force_rerun)

            tasks_by_process[process_idx].append(task)

            task_counter += 1

    logger.info(f"{'=' * 60}")
    logger.info(f"🚀 Starting parallel execution:")
    logger.info(f"   - Parallel Processes: {actual_num_processes}")
    logger.info(f"   - Available GPUs (Remapped): {remapped_gpu_ids if assigned_gpu_ids else 'CPU only'}")
    logger.info(
        f"   - Processes per GPU: {actual_num_processes / len(remapped_gpu_ids) if assigned_gpu_ids else 'N/A':.1f}")
    logger.info(f"   - Force Rerun: {force_rerun}")
    logger.info(f"{'=' * 60}")
    logger.info("--- Tasks assigned per process: ---")
    for process_id, task_list in tasks_by_process.items():
        gpu_assignment = f"GPU {process_id % len(remapped_gpu_ids)}" if assigned_gpu_ids else "CPU"
        logger.info(f"    Process {process_id} ({gpu_assignment}): {len(task_list)} tasks")
    logger.info("--- End Tasks ---")

    # Launch one process for each task list
    processes = []

    # CUDA_VISIBLE_DEVICES should already be set by setup_gpu_allocation
    if 'CUDA_VISIBLE_DEVICES' not in os.environ and gpu_ids_str:
        logger.warning("!! CUDA_VISIBLE_DEVICES not set by setup_gpu_allocation. This might fail.")
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    for process_id in range(actual_num_processes):
        task_list_for_process = tasks_by_process[process_id]
        if not task_list_for_process:
            logger.info(f"No tasks for process {process_id}, skipping.")
            continue

        p = NoDaemonProcess(target=run_task_list_serially, args=(task_list_for_process, _LOG_FILE_PATH, process_id))
        processes.append(p)
        p.start()

    # Wait for all processes to finish and check exit codes
    failed_processes = []
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            failed_processes.append((i, p.exitcode))
            logger.error(f"Process {i} exited with code {p.exitcode}")

    if failed_processes:
        logger.error(f"⚠️ {len(failed_processes)}/{len(processes)} processes failed: {failed_processes}")
    else:
        logger.info("🎉 All parallel experiments completed successfully!")


if __name__ == "__main__":
    # Set start method for CUDA compatibility
    if multiprocessing.get_start_method(allow_none=True) is None:
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(
        description="Run multiple FL experiments in parallel with caching support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        default="configs_generated",
        help="Base directory containing generated config files"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processes (will be adjusted to match GPU count if using GPUs)"
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        help="Comma-separated CUDA device IDs, e.g., '0,1,2,3'"
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Ignore cached results and rerun all experiments"
    )
    parser.add_argument(
        "--clear_results_on_rerun",  # New Flag Name
        action="store_true",
        help="[DEPRECATED by --force_rerun logic] Delete existing result files within a run directory before rerunning."
    )
    # --- END OF ADDITION ---
    args = parser.parse_args()

    main_parallel(
        configs_base_dir=args.configs_dir,
        num_processes=args.num_processes,
        gpu_ids_str=args.gpu_ids,
        force_rerun=args.force_rerun,
    )
