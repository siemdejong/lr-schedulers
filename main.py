"""Streamlit application to visualize learning rate schedulers."""

import ast
import itertools
from collections.abc import Callable, Iterable
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from torch.optim import SGD

st.set_page_config(
    page_title="LR Viz",
    page_icon=":material/schedule:",
    layout="wide",
    menu_items={
        "Report a bug": "https://www.github.com/siemdejong/lr-schedulers/issues",
    },
    initial_sidebar_state="collapsed",
)

st.title("Visualize Learning Rate Schedulers")
st.markdown(
    "This application allows you to visualize the learning rate schedule of various "
    "learning rate schedulers. The configuration options are based on the PyTorch "
    f"{version("torch").removesuffix("+cpu")} documentation. "
    "For more information, see the "
    "[PyTorch documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)."
    " See the sidebar for general settings."
)


def construct_scheduler_str(
    scheduler_cls: torch.optim.lr_scheduler.LRScheduler, parameters: dict[str, Any]
) -> str:
    """Construct a scheduler string for the given scheduler and parameters.

    Useful for showing the code block in the application.
    """
    scheduler_str = (
        "scheduler = "
        f"{scheduler_cls.__module__}.{scheduler_cls.__name__}(\n"
        "    optimizer=optimizer,\n"
    )
    for parameter in parameters:
        scheduler_str += f"    {parameter}={parameters[parameter]},\n"
    scheduler_str += ")"
    return scheduler_str


def construct_code_block(
    scheduler_cls: torch.optim.lr_scheduler.LRScheduler, parameters: dict[str, Any]
) -> str:
    """Construct a code block for the given scheduler and parameters."""
    model_str = "model = torch.nn.Linear(1, 1)"
    optimizer_str = f"optimizer = torch.optim.SGD(model.parameters(), lr={LR})"
    scheduler_str = construct_scheduler_str(scheduler_cls, parameters)
    return f"{model_str}\n{optimizer_str}\n{scheduler_str}"


def add_to_schedule_bank(
    data: pd.DataFrame, schedule_bank: pd.DataFrame
) -> pd.DataFrame:
    """Update the schedule bank with the new data."""
    # If data is a subset, don't update
    if len(data.merge(schedule_bank)) == len(data):
        return schedule_bank

    # Update schedule bank.
    current_scheduler_idx = (
        0 if len(schedule_bank) == 0 else schedule_bank["scheduler_idx"].max() + 1
    )
    data["scheduler_idx"] = current_scheduler_idx
    return pd.concat([schedule_bank, data], ignore_index=True)


def clear_schedule_bank_for_scheduler(
    scheduler_name: str, schedule_bank: pd.DataFrame
) -> pd.DataFrame:
    """Clear the schedule bank for a given scheduler."""
    max_idx = schedule_bank[schedule_bank["scheduler_name"] == scheduler_name][
        "scheduler_idx"
    ].max()
    scheduler_name_condition = schedule_bank["scheduler_name"] == scheduler_name
    scheduler_idx_condition = schedule_bank["scheduler_idx"] == max_idx
    schedule_bank = schedule_bank[scheduler_name_condition & scheduler_idx_condition]
    schedule_bank["scheduler_idx"] = 0
    return schedule_bank


def update_schedule_bank_session_state(schedule_bank: pd.DataFrame) -> None:
    """Update the schedule bank session state."""
    st.session_state["schedule_bank"] = schedule_bank


def show_schedule_bank_configs(
    scheduler_name: str, schedule_bank: pd.DataFrame
) -> None:
    """Show the configurations of the selected scheduler."""
    # We are only interested in the selected scheduler.
    data = schedule_bank[schedule_bank["scheduler_name"] == scheduler_name]

    # We don't need to display anything else than the parameters
    with st.popover("Show comparison parameters", icon=":material/compare_arrows:"):
        st.write(
            data[["scheduler_idx", "parameters"]]
            .drop_duplicates()
            .set_index("scheduler_idx")
        )


def show_lr_schedule(
    scheduler_name: str,
    scheduler_cls: torch.optim.lr_scheduler.LRScheduler,
    parameters: dict[str, Any],
    schedule_bank: pd.DataFrame,
) -> None:
    """Show the lr scheduler and the configuration.

    Paramters
    ---------
    scheduler_name : str
        The name of the scheduler.
    scheduler_cls : torch.optim.lr_scheduler.LRScheduler
        The scheduler class.
    """
    plot_col, config_col = st.columns([0.6, 0.4])

    with config_col:
        selected_parameters = show_config_inputs(scheduler_name, parameters)

    data = calc_data(scheduler_cls, selected_parameters)
    schedule_bank = add_to_schedule_bank(data, schedule_bank)
    update_session_state = config_col.button(
        "Save",
        key="add_to_schedule_bank" + scheduler_name,
        icon=":material/bookmark:",
    )
    if update_session_state:
        update_schedule_bank_session_state(schedule_bank)
    clear_schedules = config_col.button(
        "Clear",
        key="clear_schedule_bank" + scheduler_name,
        icon=":material/delete_forever:",
    )
    if clear_schedules:
        schedule_bank = clear_schedule_bank_for_scheduler(scheduler_name, schedule_bank)
        update_schedule_bank_session_state(schedule_bank)
    fig = plot_schedules(schedule_bank=schedule_bank, filter_scheduler=scheduler_name)
    plot_col.plotly_chart(fig)

    with config_col.popover("Show code", icon=":material/code:"):
        code_block = construct_code_block(scheduler_cls, selected_parameters)
        st.code(code_block)

    with config_col:
        show_schedule_bank_configs(scheduler_name, schedule_bank)


with st.sidebar:
    st.markdown("# Configuration")
    STEPS = st.number_input("Number of steps", value=100, min_value=0, max_value=1000)
    LR = st.number_input("Learning rate given to optimizer", value=0.1)
    PLOT_MAX_LR = st.number_input("Maximum learning rate for plot", value=LR)


def calc_data(
    scheduler_cls: torch.optim.lr_scheduler.LRScheduler,
    parameters: dict[str, Any],
) -> dict:
    """Calculate the learning rate schedule."""
    optimizer = SGD([torch.tensor(1)], lr=LR)
    scheduler = scheduler_cls(optimizer, **parameters)
    data = {"step": np.zeros(STEPS), "lr": np.zeros(STEPS)}

    for step in range(STEPS):
        optimizer.step()
        data["step"][step] = step
        data["lr"][step] = scheduler.get_last_lr()[0]
        scheduler.step()

    return pd.DataFrame(
        {
            "scheduler_name": scheduler_cls.__name__,
            "lr": data["lr"],
            "step": data["step"],
            "parameters": str(parameters),
        }
    )


def plot_schedules(
    schedule_bank: pd.DataFrame,
    filter_scheduler: str | None = None,
) -> go.Figure:
    """Plot the learning rate schedule for a given scheduler."""
    if filter_scheduler is not None:
        schedule_bank = schedule_bank[
            schedule_bank["scheduler_name"] == filter_scheduler
        ]
    return px.line(
        schedule_bank,
        x="step",
        y="lr",
        color="scheduler_idx",
        title=filter_scheduler if filter_scheduler is not None else "All schedulers",
        range_x=[0, STEPS],
        range_y=[0, PLOT_MAX_LR],
    )


def check_lambda_for_safety(lambda_str: str) -> str:
    """Sanitize the lambda string."""
    if not lambda_str.startswith("lambda x:") and lambda_str != "None":
        error_msg = "Lambda function should start with 'lambda x:'."
        raise ValueError(error_msg)
    allowed_chars = "x.,*/-+0123456789() "
    allowed_functions = {"x", "min", "max", "abs"}
    lambda_body = lambda_str.replace("lambda x:", "").replace("None", "").strip()

    # Check for invalid characters
    if not all(
        char in allowed_chars
        for char in lambda_body
        if char.isalnum() or char in allowed_chars
    ):
        error_msg = (
            f"Lambda function ({lambda_body}) contains invalid characters. "
            "Please only use 'x', '/', '*', '-', '+', '0-9', spaces, and allowed words."
        )
        raise ValueError(error_msg)

    # Check for allowed words
    tokens = lambda_body.split()
    for token in tokens:
        if token.isalpha() and token not in allowed_functions:
            error_msg = (
                f"Lambda function contains invalid word '{token}'. "
                f"Allowed words are {allowed_functions}."
            )
            raise ValueError(error_msg)

    return lambda_str


def show_config_inputs(
    scheduler_name: str,
    parameters: dict[str, Any],
) -> int | float | bool | Iterable | Callable | str:
    """Show configuration options for a given scheduler."""
    selected_parameters = {}

    left_column, _, right_column = st.columns([0.48, 0.04, 0.48])

    for parameter, column in zip(
        parameters, itertools.cycle([left_column, right_column])
    ):
        _type = type(parameters[parameter])
        if _type is int:
            if (
                scheduler_name == "CosineAnnealingWarmRestarts"
                and parameter == "T_mult"
            ):
                min_value = 1
            elif scheduler_name == "OneCycleLR" and parameter == "total_steps":
                min_value = STEPS
            else:
                min_value = None
            selected_parameters[parameter] = column.number_input(
                parameter,
                value=parameters[parameter],
                min_value=min_value,
                key=scheduler_name + "_" + parameter,
            )
        elif _type is float:
            if scheduler_name in ["ConstantLR", "LinearLR"] and "factor" in parameter:
                selected_parameters[parameter] = column.slider(
                    parameter,
                    min_value=1e-9,
                    max_value=1.0,
                    value=parameters[parameter],
                    key=scheduler_name + "_" + parameter,
                )
            else:
                if (
                    scheduler_name == "CosineAnnealingWarmRestarts"
                    and parameter == "T_0"
                ):
                    min_value = 0.0
                elif scheduler_name == "OneCycleLR" and parameter == "pct_start":
                    min_value = 0.0
                    max_value = 1.0
                else:
                    min_value = None
                    max_value = None
                selected_parameters[parameter] = column.number_input(
                    parameter,
                    value=parameters[parameter],
                    step=1e-5,
                    min_value=min_value,
                    max_value=max_value,
                    format="%.5f",
                    key=scheduler_name + "_" + parameter,
                )
        elif _type is bool:
            selected_parameters[parameter] = column.checkbox(
                parameter,
                value=parameters[parameter],
                key=scheduler_name + "_" + parameter,
            )
        elif _type is tuple:
            options: list[Any] = parameters[parameter][1]
            default_idx = options.index(parameters[parameter][0])
            selected_parameters[parameter] = column.radio(
                parameter,
                options=options,
                index=default_idx,
                key=scheduler_name + "_" + parameter,
            )
        elif _type is list:
            selected_parameters[parameter] = ast.literal_eval(
                column.text_input(
                    parameter,
                    parameters[parameter],
                    key=scheduler_name + "_" + parameter,
                )
            )
        elif _type is str and (
            parameters[parameter].startswith("lambda")
            or (
                parameters[parameter].startswith("None")
                and scheduler_name == "CyclicLR"
            )
        ):
            try:
                eval_str = check_lambda_for_safety(
                    column.text_input(
                        parameter,
                        parameters[parameter],
                        key=scheduler_name + "_" + parameter,
                    )
                )
            except ValueError as e:
                column.error(e)
                selected_parameters[parameter] = eval(parameters[parameter])  # noqa: S307, default is safe.
            else:
                selected_parameters[parameter] = eval(  # noqa: S307
                    eval_str
                )

    return selected_parameters


def main() -> None:
    """Entrypoint for application."""
    torch_optim_lr_schedulers = {
        "ConstantLR": {
            "cls": torch.optim.lr_scheduler.ConstantLR,
            "factor": 0.33,
            "total_iters": STEPS // 2,
        },
        "CosineAnnealingLR": {
            "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
            "T_max": STEPS,
            "eta_min": 0.0,
        },
        "CosineAnnealingWarmRestarts": {
            "cls": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            "T_0": 20,
            "T_mult": 1,
            "eta_min": 0.0,
        },
        "CyclicLR": {
            "cls": torch.optim.lr_scheduler.CyclicLR,
            "base_lr": 0,
            "max_lr": 0.1,
            "step_size_up": 10,
            "mode": ("triangular", ("triangular", "triangular2", "exp_range")),
            "gamma": 0.95,
            "scale_fn": "None",
            "scale_mode": ("cycle", ("cycle", "iterations")),
            # TODO(siemdejong): Allow for custom functions.
            # https://github.com/siemdejong/lr-schedulers/issues/1
            # TODO(siemdejong): Allow for momentum scheduling.
            # https://github.com/siemdejong/lr-schedulers/issues/2
        },
        "ExponentialLR": {
            "cls": torch.optim.lr_scheduler.ExponentialLR,
            "gamma": 0.95,
        },
        "LambdaLR": {
            "cls": torch.optim.lr_scheduler.LambdaLR,
            "lr_lambda": "lambda x: 0.95 ** x",
        },
        "LinearLR": {
            "cls": torch.optim.lr_scheduler.LinearLR,
            "start_factor": 0.33,
            "end_factor": 1.0,
            "total_iters": STEPS // 2,
        },
        "MultiStepLR": {
            "cls": torch.optim.lr_scheduler.MultiStepLR,
            "milestones": [33, 66],
            "gamma": 0.1,
        },
        "MultiplicativeLR": {
            "cls": torch.optim.lr_scheduler.MultiplicativeLR,
            "lr_lambda": "lambda x: 0.95",
        },
        "OneCycleLR": {
            "cls": torch.optim.lr_scheduler.OneCycleLR,
            "total_steps": STEPS,
            "max_lr": 0.1,
            "pct_start": 0.3,
            "anneal_strategy": ("cos", ("cos", "linear")),
            "div_factor": 25.0,
            "final_div_factor": 10000.0,
        },
        "PolynomialLR": {
            "cls": torch.optim.lr_scheduler.PolynomialLR,
            "total_iters": STEPS,
            "power": 1,
        },
        "StepLR": {
            "cls": torch.optim.lr_scheduler.StepLR,
            "step_size": 33,
            "gamma": 0.1,
        },
    }

    if "schedule_bank" not in st.session_state:
        st.session_state["schedule_bank"] = pd.DataFrame(
            columns=["scheduler_name", "lr", "step", "parameters", "scheduler_idx"]
        )
    schedule_bank = st.session_state["schedule_bank"]

    for lr_scheduler in torch_optim_lr_schedulers:
        st.divider()
        st.fragment(
            show_lr_schedule(
                lr_scheduler,
                torch_optim_lr_schedulers[lr_scheduler].pop("cls"),
                torch_optim_lr_schedulers[lr_scheduler],
                schedule_bank=schedule_bank,
            )
        )


if __name__ == "__main__":
    main()
