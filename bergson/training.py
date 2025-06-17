from transformers import TrainerCallback


class GradientCollectorCallback(TrainerCallback):
    """
    Callback to collect document-level gradients during training.
    """

    def __init__(self, processor):
        self.processor = processor

    def on_step_end(self, args, state, control, **kwargs):
        """
        Collect gradients at the end of each step.
        """
        if hasattr(state, "model") and state.model is not None:
            self.processor.collect_gradients(state.model)
