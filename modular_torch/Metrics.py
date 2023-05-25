import torch

class AccuracyFn:
    def __init__(self):
        """normal accuracy function
        """
        pass
           
    def __call__(self, y_logits: torch.Tensor, y_true:torch.Tensor) -> float:
        """_summary_

        Args:
            y_logits (torch.Tensor): predicted tensor
            y_true (torch.Tensor): true ground truth tensor

        Returns:
            float: accuracy of predicted tensor.
        """
        
        y_probs = torch.argmax(torch.softmax(y_logits, dim = 1), dim = 1)
        return ((y_probs == y_true).sum().item() / len(y_probs))

    def _get_name(self):
        return "accuracy"