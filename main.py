"""Streamlit application to visualize learning rate schedulers."""

import ast
import itertools
from collections.abc import Callable, Iterable
from importlib.metadata import version
from typing import Any

import numpy as np
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

TORCH_VERSION = version("torch").removesuffix("+cpu")

st.title("Visualize Learning Rate Schedulers")
st.markdown(
    "This application allows you to visualize the learning rate schedule of various "
    "learning rate schedulers. The configuration options are based on the PyTorch "
    f"{TORCH_VERSION} documentation. For more information, see the "
    "[PyTorch documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)."
    " See the sidebar for general settings."
)


def show_lr_schedule(
    scheduler_name: str,
    scheduler_cls: torch.optim.lr_scheduler.LRScheduler,
    parameters: dict[str, Any],
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
    selected_parameters = show_config(scheduler_name, parameters, config_col)

    with plot_col:
        fig = plot_schedule(scheduler_name, scheduler_cls, selected_parameters)
        st.plotly_chart(fig)


with st.sidebar:
    st.markdown("# Configuration")
    STEPS = st.number_input("Number of steps", value=100, min_value=0, max_value=1000)
    LR = st.number_input("Learning rate given to optimizer", value=0.1)
    PLOT_MAX_LR = st.number_input("Maximum learning rate for plot", value=LR)


def calc_data(
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler
) -> dict:
    """Calculate the learning rate schedule."""
    data = {"step": np.zeros(STEPS), "lr": np.zeros(STEPS)}
    for step in range(STEPS):
        optimizer.step()
        data["step"][step] = step
        data["lr"][step] = scheduler.get_last_lr()[0]
        scheduler.step()
    return data


def plot_schedule(
    scheduler_name: str,
    scheduler_cls: torch.optim.lr_scheduler.LRScheduler,
    parameters: dict,
) -> go.Figure:
    """Plot the learning rate schedule for a given scheduler."""
    optimizer = SGD([torch.tensor(1)], lr=LR)

    try:
        scheduler = scheduler_cls(optimizer, **parameters)
        data = calc_data(optimizer, scheduler)
    except ValueError as e:
        st.error(e)
        data = {"step": [], "lr": []}
    finally:
        fig = px.line(
            data,
            x="step",
            y="lr",
            title=scheduler_name,
            range_x=[0, STEPS],
            range_y=[0, PLOT_MAX_LR],
        )
    return fig


def show_config(
    scheduler_name: str,
    parameters: dict[str, Any],
    st_container: st.container = st.container,
) -> int | float | bool | Iterable | Callable | str:
    """Show configuration options for a given scheduler."""
    selected_parameters = {}

    left_column, _, right_column = st_container.columns([0.48, 0.04, 0.48])

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
                    min_value=0.0,
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
            # TODO(siemdejong): Allow for custom functions.
            # https://github.com/siemdejong/lr-schedulers/issues/1
            # TODO(siemdejong): Allow for momentum scheduling.
            # https://github.com/siemdejong/lr-schedulers/issues/2
        },
        "ExponentialLR": {
            "cls": torch.optim.lr_scheduler.ExponentialLR,
            "gamma": 0.95,
        },
        # TODO(siemdejong): implement LambdaLR
        # https://github.com/siemdejong/lr-schedulers/issues/1
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
        # TODO(siemdejong): implement MultiplicativeLR
        # https://github.com/siemdejong/lr-schedulers/issues/1
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
    for lr_scheduler in torch_optim_lr_schedulers:
        st.divider()
        st.fragment(
            show_lr_schedule(
                lr_scheduler,
                torch_optim_lr_schedulers[lr_scheduler].pop("cls"),
                torch_optim_lr_schedulers[lr_scheduler],
            )
        )


main()
