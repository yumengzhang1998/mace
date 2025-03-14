from tkinter import E
from typing import Dict, Tuple, Optional, List,Any
from mace.tools.scripts_utils import SubsetCollection
from mace.tools.multihead_tools import HeadConfig
from torch_geometric.data import Data
from mace import data, modules, tools
import dataclasses
import argparse
from mace.tools import AtomicNumberTable
import json
import numpy as np
def build_default_arg_parser_dict(defaults: dict = None) -> dict:
    """
    Builds a default dictionary-based argument parser.

    :param defaults: Optional dictionary containing default argument values.
    :return: Dictionary with parsed arguments.
    """
    defaults = defaults or {}

    args_dict = {
        "name": defaults.get("name", None),  # Required
        "seed": defaults.get("seed", 123),
        "work_dir": defaults.get("work_dir", "."),
        "log_dir": defaults.get("log_dir", None),
        "model_dir": defaults.get("model_dir", None),
        "checkpoints_dir": defaults.get("checkpoints_dir", None),
        "results_dir": defaults.get("results_dir", None),
        "downloads_dir": defaults.get("downloads_dir", None),
        "device": defaults.get("device", "cpu"),
        "default_dtype": defaults.get("default_dtype", "float64"),
        "distributed": defaults.get("distributed", False),
        "log_level": defaults.get("log_level", "INFO"),
        "error_table": defaults.get("error_table", "PerAtomRMSE"),
        "model": defaults.get("model", "MACE"),
        "r_max": defaults.get("r_max", 5.0),
        "radial_type": defaults.get("radial_type", "bessel"),
        "num_radial_basis": defaults.get("num_radial_basis", 8),
        "num_cutoff_basis": defaults.get("num_cutoff_basis", 5),
        "pair_repulsion": defaults.get("pair_repulsion", False),
        "distance_transform": defaults.get("distance_transform", "None"),
        "interaction": defaults.get("interaction", "RealAgnosticResidualInteractionBlock"),
        "interaction_first": defaults.get("interaction_first", "RealAgnosticResidualInteractionBlock"),
        "max_ell": defaults.get("max_ell", 3),
        "correlation": defaults.get("correlation", 3),
        "num_interactions": defaults.get("num_interactions", 2),
        "MLP_irreps": defaults.get("MLP_irreps", "16x0e"),
        "radial_MLP": defaults.get("radial_MLP", "[64, 64, 64]"),
        "hidden_irreps": defaults.get("hidden_irreps", None),
        "num_channels": defaults.get("num_channels", None),
        "max_L": defaults.get("max_L", None),
        "gate": defaults.get("gate", "silu"),
        "scaling": defaults.get("scaling", "rms_forces_scaling"),
        "avg_num_neighbors": defaults.get("avg_num_neighbors", 1),
        "compute_avg_num_neighbors": defaults.get("compute_avg_num_neighbors", True),
        "compute_stress": defaults.get("compute_stress", False),
        "compute_forces": defaults.get("compute_forces", True),
        "train_file": defaults.get("train_file", None),
        "valid_file": defaults.get("valid_file", None),
        "valid_fraction": defaults.get("valid_fraction", 0.1),
        "test_file": defaults.get("test_file", None),
        "test_dir": defaults.get("test_dir", None),
        "multi_processed_test": defaults.get("multi_processed_test", False),
        "num_workers": defaults.get("num_workers", 0),
        "pin_memory": defaults.get("pin_memory", True),
        "atomic_numbers": defaults.get("atomic_numbers", None),
        "mean": defaults.get("mean", None),
        "std": defaults.get("std", None),
        "statistics_file": defaults.get("statistics_file", None),
        "E0s": defaults.get("E0s", None),
        "foundation_filter_elements": defaults.get("foundation_filter_elements", True),
        "heads": defaults.get("heads", None),
        "multiheads_finetuning": defaults.get("multiheads_finetuning", True),
        "foundation_head": defaults.get("foundation_head", None),
        "weight_pt_head": defaults.get("weight_pt_head", 1.0),
        "num_samples_pt": defaults.get("num_samples_pt", 10000),
        "force_mh_ft_lr": defaults.get("force_mh_ft_lr", False),
        "subselect_pt": defaults.get("subselect_pt", "random"),
        "pt_train_file": defaults.get("pt_train_file", None),
        "pt_valid_file": defaults.get("pt_valid_file", None),
        "foundation_model_elements": defaults.get("foundation_model_elements", False),
        "keep_isolated_atoms": defaults.get("keep_isolated_atoms", False),
        "energy_key": defaults.get("energy_key", "REF_energy"),
        "forces_key": defaults.get("forces_key", "REF_forces"),
        "virials_key": defaults.get("virials_key", "REF_virials"),
        "stress_key": defaults.get("stress_key", "REF_stress"),
        "dipole_key": defaults.get("dipole_key", "REF_dipole"),
        "charges_key": defaults.get("charges_key", "REF_charges"),
        "loss": defaults.get("loss", "weighted"),
        "forces_weight": defaults.get("forces_weight", 100.0),
        "energy_weight": defaults.get("energy_weight", 1.0),
        "virials_weight": defaults.get("virials_weight", 1.0),
        "stress_weight": defaults.get("stress_weight", 1.0),
        "dipole_weight": defaults.get("dipole_weight", 1.0),
        "config_type_weights": defaults.get("config_type_weights", '{"Default":1.0}'),
        "huber_delta": defaults.get("huber_delta", 0.01),
        "optimizer": defaults.get("optimizer", "adam"),
        "beta": defaults.get("beta", 0.9),
        "batch_size": defaults.get("batch_size", 10),
        "valid_batch_size": defaults.get("valid_batch_size", 10),
        "lr": defaults.get("lr", 0.01),
        "weight_decay": defaults.get("weight_decay", 5e-7),
        "amsgrad": defaults.get("amsgrad", True),
        "scheduler": defaults.get("scheduler", "ReduceLROnPlateau"),
        "lr_factor": defaults.get("lr_factor", 0.8),
        "scheduler_patience": defaults.get("scheduler_patience", 50),
        "lr_scheduler_gamma": defaults.get("lr_scheduler_gamma", 0.9993),
        "ema": defaults.get("ema", False),
        "ema_decay": defaults.get("ema_decay", 0.99),
        "max_num_epochs": defaults.get("max_num_epochs", 2048),
        "patience": defaults.get("patience", 2048),
        "foundation_model": defaults.get("foundation_model", None),
        "foundation_model_readout": defaults.get("foundation_model_readout", True),
        "eval_interval": defaults.get("eval_interval", 1),
        "keep_checkpoints": defaults.get("keep_checkpoints", False),
        "save_all_checkpoints": defaults.get("save_all_checkpoints", False),
        "restart_latest": defaults.get("restart_latest", False),
        "save_cpu": defaults.get("save_cpu", False),
        "clip_grad": defaults.get("clip_grad", 10.0),
        "enable_cueq": defaults.get("enable_cueq", False),
        "wandb": defaults.get("wandb", False),
        "wandb_dir": defaults.get("wandb_dir", None),
        "wandb_project": defaults.get("wandb_project", ""),
        "wandb_entity": defaults.get("wandb_entity", ""),
        "wandb_name": defaults.get("wandb_name", ""),
        "swa": defaults.get("swa", False),
        "start_swa": defaults.get("start_swa", None),
        "swa_lr": defaults.get("swa_lr", 1e-3),
        "swa_energy_weight": defaults.get("swa_energy_weight", 1000.0),
        "swa_forces_weight": defaults.get("swa_forces_weight", 100.0),
        "swa_virials_weight": defaults.get("swa_virials_weight", 10.0),
        "swa_stress_weight": defaults.get("swa_stress_weight", 10.0),
        "swa_dipole_weight": defaults.get("swa_dipole_weight", 1.0),
    }

    args = argparse.Namespace(**args_dict) 
    return args



def get_dataset_from_data_objects_multihead(
    data_dict: Dict[str, Dict[str, Optional[list]]]
) -> Dict[str, HeadConfig]:
    """
    Accepts a dictionary where each key is a head name, and values contain train/valid/test Data objects.
    
    Args:
        data_dict (Dict[str, Dict[str, Optional[list]]]): 
            - Key: head_name (str)
            - Value: {
                "train": List of Data objects,
                "valid": List of Data objects (optional),
                "test": List of (name, List of Data) tuples (optional)
            }

    Returns:
        Dict[str, HeadConfig]: Mapping of head names to configured HeadConfig objects.
    """
    head_configs = {}

    for head_name, datasets in data_dict.items():
        train_data = datasets.get("train", [])
        valid_data = datasets.get("valid", [])
        test_data = datasets.get("test", [])

        # Store the dataset in a SubsetCollection
        collections = SubsetCollection(
            train=train_data,
            valid=valid_data if valid_data else [],
            tests=test_data if test_data else [],
        )

        # Create a HeadConfig object for this head
        head_config = HeadConfig(
            head_name=head_name,
            collections=collections,
            atomic_energies_dict=None,  # Define if needed
        )

        head_configs[head_name] = head_config

    return head_configs
def convert_data_to_configuration(data_list: List[Data]) -> data.Configurations:
    """
    Convert a list of torch_geometric Data objects into MACE Configuration objects.
    """
    configurations = []
    for d in data_list:
        config = data.Configuration(
            atomic_numbers=d.z.numpy(),
            positions=d.pos.numpy(),
            energy=d.y.item() if d.y is not None else None,
            forces=d.forces.numpy() if d.forces is not None else None,
            charges=d.charge if d.charge is not None else None,
            config_type="Default",
            head="Default",
        )
        configurations.append(config)
    return configurations

def get_dataset_from_xyz_variable(
    train_list: List[Data],
    valid_list: Optional[List[Data]] = None,
    valid_fraction: float = 0.1,
    seed: int = 1234,
    config_type_weights: Optional[Dict] = None,
    head_name: str = "Default",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """
    Load dataset from lists of Data objects instead of a file,
    ensuring the same structure as get_dataset_from_xyz.

    Args:
        train_list: List of Data objects for training.
        valid_list: Optional list of Data objects for validation. If None, split from train_list.
        valid_fraction: Fraction of data to use for validation if valid_list is None.
        seed: Random seed for reproducibility.
        config_type_weights: Optional configuration type weights.
        head_name: Name of the head.

    Returns:
        Tuple containing SubsetCollection with train, validation, and test sets, along with optional atomic energies dictionary.
    """
    train_configs = convert_data_to_configuration(train_list)
    
    if valid_list is None:
        train_configs, valid_configs = data.random_train_valid_split(
            train_configs, valid_fraction, seed, "work_dir"
        )
    else:
        valid_configs = convert_data_to_configuration(valid_list)
    
    atomic_energies_dict = {}
    
    return SubsetCollection(train=train_configs, valid=valid_configs, tests=[]), atomic_energies_dict

def get_atomic_energies_from_data(E0s, train, z_table) -> dict:
    """
    train is a list of Data objects
    """
    if isinstance(E0s, str) and E0s.lower() == "average":
        logging.info("Computing average Atomic Energies using least squares regression")
        try:
            assert train is not None
            atomic_energies_dict = compute_average_E0s(train, z_table)
        except Exception as e:
            raise RuntimeError(
                f"Could not compute average E0s if no training xyz given, error {e} occurred"
            ) from e
    
    # If E0s is provided as a JSON file, load atomic energies from it
    elif isinstance(E0s, str) and E0s.endswith(".json"):
        logging.info(f"Loading atomic energies from {E0s}")
        try:
            with open(E0s, "r", encoding="utf-8") as f:
                atomic_energies_dict = json.load(f)
                atomic_energies_dict = {int(key): value for key, value in atomic_energies_dict.items()}
        except Exception as e:
            raise RuntimeError(f"Failed to load E0s from JSON file: {e}") from e
    
    # If E0s is provided as a dictionary or string representation of a dictionary, parse it
    elif E0s is not None:
        try:
            if isinstance(E0s, str):
                atomic_energies_dict = ast.literal_eval(E0s)
            elif isinstance(E0s, dict):
                atomic_energies_dict = E0s
            else:
                raise ValueError("E0s must be a string, dictionary, or a valid JSON file path")
            
            assert isinstance(atomic_energies_dict, dict)
        except Exception as e:
            raise RuntimeError(f"E0s specified invalidly, error {e} occurred") from e
    else:
        raise RuntimeError(
            "E0s not found in training file and not specified in command line"
        )
    return atomic_energies_dict



def compute_average_E0s(
    train: List, z_table: AtomicNumberTable
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = train[i].y
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(train[i].z == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.error(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict

@dataclasses.dataclass
class HeadConfig:
    head_name: str
    train_data: Optional[list] = None  # List of Data objects (torch_geometric)
    valid_data: Optional[list] = None  # Validation Data
    test_data: Optional[list] = None   # Test Data
    statistics_file: Optional[str] = None
    config_type_weights: Optional[Dict[str, float]] = None
    energy_key: Optional[str] = "y"  # Key for energy values
    forces_key: Optional[str] = "forces"  # Key for force values
    atomic_numbers: Optional[List[int]] = None
    compute_avg_num_neighbors: Optional[bool] = False
    z_table: Optional[Any] = None
    E0s: Optional[Any] = None


### configure model without foundation model
import ast
import logging
from e3nn import o3
from mace import modules
from mace.tools.finetuning_utils import load_foundations_elements

def configure_model_without_scaleshift(
    args, atomic_energies, model_foundation=None, heads=None, z_table=None
):
    # Selecting outputs
    compute_virials = args.loss in ("stress", "virials", "huber", "universal")
    if compute_virials:
        args.compute_stress = True
        args.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": args.compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": args.compute_dipole,
    }
    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )
    logging.info("===========MODEL DETAILS===========")



    # Build model
    logging.info("Building model")
    logging.info(
        f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
    )
    logging.info(
        f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
    )
    logging.info(
        f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
    )
    logging.info(
        f"Radial cutoff: {args.r_max} A (total receptive field for each atom: {args.r_max * args.num_interactions} A)"
    )
    logging.info(
        f"Distance transform for radial basis functions: {args.distance_transform}"
    )

    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

    logging.info(f"Hidden irreps: {args.hidden_irreps}")

    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
    )
    model_config_foundation = None

    # model = _build_model(args, model_config, model_config_foundation, heads)

    return model_config, output_args
    # return model, output_args


def _determine_atomic_inter_shift(mean, heads):
    if isinstance(mean, np.ndarray):
        if mean.size == 1:
            return mean.item()
        if mean.size == len(heads):
            return mean.tolist()
        logging.info("Mean not in correct format, using default value of 0.0")
        return [0.0] * len(heads)
    if isinstance(mean, list) and len(mean) == len(heads):
        return mean
    if isinstance(mean, float):
        return [mean] * len(heads)
    logging.info("Mean not in correct format, using default value of 0.0")
    return [0.0] * len(heads)


def _build_model(
    args, model_config, model_config_foundation, heads
):  # pylint: disable=too-many-return-statements
    if args.model == "MACE":
        if args.interaction_first not in [
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
        ]:
            args.interaction_first = "RealAgnosticInteractionBlock"
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == "MACE_with_charge":
        return modules.ScaleShiftMACE_with_charge(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == 'MACE_with_charge_after':
        return modules.ScaleShiftMACE_with_charge_after(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == "ScaleShiftMACE":
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == "FoundationMACE":
        return modules.ScaleShiftMACE(**model_config_foundation)
    if args.model == "ScaleShiftBOTNet":
        return modules.ScaleShiftBOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
        )
    if args.model == "BOTNet":
        return modules.BOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    # if args.model == "AtomicDipolesMACE":
    #     assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
    #     assert (
    #         args.error_table == "DipoleRMSE"
    #     ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
    #     return modules.AtomicDipolesMACE(
    #         **model_config,
    #         correlation=args.correlation,
    #         gate=modules.gate_dict[args.gate],
    #         interaction_cls_first=modules.interaction_classes[
    #             "RealAgnosticInteractionBlock"
    #         ],
    #         MLP_irreps=o3.Irreps(args.MLP_irreps),
    #     )
    # if args.model == "EnergyDipolesMACE":
    #     assert (
    #         args.loss == "energy_forces_dipole"
    #     ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
    #     assert (
    #         args.error_table == "EnergyDipoleRMSE"
    #     ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
    #     return modules.EnergyDipolesMACE(
    #         **model_config,
    #         correlation=args.correlation,
    #         gate=modules.gate_dict[args.gate],
    #         interaction_cls_first=modules.interaction_classes[
    #             "RealAgnosticInteractionBlock"
    #         ],
    #         MLP_irreps=o3.Irreps(args.MLP_irreps),
    #     )
    raise RuntimeError(f"Unknown model: '{args.model}'")

