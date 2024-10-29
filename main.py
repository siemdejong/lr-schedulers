"""Streamlit application to visualize learning rate schedulers."""

import ast
import inspect
import typing
from collections.abc import Callable, Iterable
from collections.abc import Callable as TCallable
from typing import _CallableGenericAlias, get_args, get_origin

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from torch.optim import SGD

STEPS = 100

DEFAULTS = {
    "CosineAnnealingLR": {"T_max": STEPS},
    "CosineAnnealingWarmRestarts": {"T_0": STEPS // 5},
    "CyclicLR": {"base_lr": 0.001, "max_lr": 0.1, "step_size_up": STEPS // 5},
    "ExponentialLR": {"gamma": 0.95},
    "LambdaLR": {"lr_lambda": "lambda x: 0.95 ** x"},
    "MultiStepLR": {"milestones": [STEPS // 3, 2 * STEPS // 3], "gamma": 0.1},
    "MultiplicativeLR": {"lr_lambda": "lambda x: 0.95"},
    "OneCycleLR": {"max_lr": 0.1},
    "StepLR": {"step_size": STEPS // 3},
}

Callable = Callable | TCallable | _CallableGenericAlias


@st.cache_data
def plot_schedule(
    scheduler_name: str,
    scheduler_cls: torch.optim.lr_scheduler.LRScheduler,
    parameters: dict,
) -> go.Figure:
    """Plot the learning rate schedule for a given scheduler."""
    optimizer = SGD([torch.tensor(1)], lr=1)
    # Use a scheduler of your choice below.
    # Great for debugging your own schedulers!
    if inspect.signature(scheduler_cls).parameters.get("total_iters"):
        total_iters_scheduler_kwargs = {"total_iters": STEPS}
    elif inspect.signature(scheduler_cls).parameters.get("total_steps"):
        total_iters_scheduler_kwargs = {"total_steps": STEPS}
    else:
        total_iters_scheduler_kwargs = {}

    defaults = DEFAULTS.get(scheduler_name, {}) | parameters

    # Convert lambdas strings to functions
    for parameter in defaults:
        if "lambda" in str(defaults[parameter]):
            st.write(f"Converting {defaults[parameter]} to function")
            defaults[parameter] = ast.literal_eval(defaults[parameter])

    scheduler_kwargs = defaults | total_iters_scheduler_kwargs
    scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    data = {"step": np.zeros(STEPS), "lr": np.zeros(STEPS)}
    for step in range(STEPS):
        optimizer.step()
        data["step"][step] = step
        data["lr"][step] = scheduler.get_last_lr()[0]
        scheduler.step()

    fig = px.line(data, x="step", y="lr", title=scheduler_name)
    fig.update_yaxes(range=[0, None])
    return fig


def show_config(  # noqa: PLR0912, C901
    scheduler_name: str,
    scheduler_cls: torch.optim.lr_scheduler.LRScheduler,
) -> int | float | bool | Iterable | Callable | str:
    """Show configuration options for a given scheduler."""
    signature = inspect.signature(scheduler_cls)
    selected_inputs = {}
    for parameter_name in signature.parameters:
        if parameter_name in ["optimizer", "last_epoch", "verbose", "total_iters"]:
            continue

        parameter = signature.parameters[parameter_name]
        annotation = parameter.annotation
        default = None
        if annotation is inspect.Signature.empty:
            annotation = type(parameter.default)
            default = parameter.default
        elif get_origin(annotation) == typing.Union:
            annotation = get_args(parameter.annotation)[0]
            default = parameter.default
        elif get_origin(annotation) is typing.Literal:
            annotation = get_args(parameter.annotation)
            default = parameter.default
        elif get_origin(annotation) is bool:
            annotation = bool
            default = parameter.default
        elif get_origin(annotation) is Iterable:
            annotation = Iterable
            default = parameter.default
        elif annotation is int:
            annotation = int
            default = parameter.default
        elif annotation is float:
            annotation = float
            default = parameter.default
        elif annotation is Callable or type(annotation) is _CallableGenericAlias:
            annotation = Callable
            default = parameter.default

        if default in [inspect.Signature.empty, None]:
            defaults = DEFAULTS.get(scheduler_name)
            default = defaults.get(str(parameter.name)) if defaults else None

        if annotation in (int, float):
            selected_input = st.slider(
                f"{parameter_name}",
                value=default,
                key=f"{scheduler_name}-{parameter_name}",
            )
        elif type(annotation) is tuple:
            selected_input = st.radio(
                f"{parameter_name}",
                options=annotation,
                index=annotation.index(default),
                key=f"{scheduler_name}-{parameter_name}",
            )
        elif annotation is bool:
            selected_input = st.toggle(
                f"{parameter_name}",
                value=default,
                key=f"{scheduler_name}-{parameter_name}",
            )
        elif annotation is Iterable:
            selected_input = st.text_input(
                f"{parameter_name}",
                key=f"{scheduler_name}-{parameter_name}",
                value=default,
            )
            selected_input = ast.literal_eval(selected_input)
        elif annotation is Callable or type(annotation) is _CallableGenericAlias:
            selected_input = st.text_input(f"{parameter_name}", value=default)
        else:
            selected_input = st.text_input(
                f"{parameter_name} (not recognized)",
                disabled=True,
                value=default,
            )
        selected_inputs[parameter_name] = selected_input

    return selected_inputs


def main() -> None:
    """Entrypoint for application."""
    st.set_page_config(
        page_title="LR Viz",
        page_icon=":material/schedule:",
        layout="wide",
        menu_items={
            "Report a bug": "https://www.github.com/siemdejong/lr-schedulers/issues",
        }
    )

    st.title("Visualize Learning Rate Schedulers")

    torch_optim_lr_scheduler_members = sorted(
        inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass),
        key=lambda x: x[0],
    )
    for member_name, member_cls in torch_optim_lr_scheduler_members:
        if issubclass(
            member_cls,
            torch.optim.lr_scheduler.LRScheduler,
        ) and member_name not in [
            "_LRScheduler",
            "LRScheduler",
            "ChainedScheduler",
            "SequentialLR",
            "ReduceLROnPlateau",
        ]:
            plot_col, config_col = st.columns([0.7, 0.3])

            with config_col:
                parameters = show_config(member_name, member_cls)

            with plot_col:
                fig = plot_schedule(member_name, member_cls, parameters)
                fig.update()
                st.plotly_chart(fig)

            st.divider()


if __name__ == "__main__":
    main()
