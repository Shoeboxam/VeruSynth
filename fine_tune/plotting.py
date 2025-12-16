from typing import List
from transformers import TrainerCallback, TrainingArguments
import matplotlib.pyplot as plt


class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_losses: List[float] = []
        self.train_steps: List[int] = []

    def on_log(self, args: TrainingArguments, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(state.global_step)


def plot_loss(history: LossHistoryCallback):
    if not history.train_steps:
        print("No loss history recorded.")
        return

    plt.figure()
    plt.plot(history.train_steps, history.train_losses, marker="o")
    plt.xlabel("Global step")
    plt.ylabel("Training loss")
    plt.title("Training loss over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

