import torch

def compute_response_loss(student_outputs, teacher_outputs, temprature=1.0):
    """
    Computes the response-based knowledge distillation loss between the student and teacher model outputs.
    This function calculates the loss by comparing the softened logits of the student model to the softened logits of the teacher model. 
    The logits are softened using the softmax function and a temperature parameter, as described in the paper 
    "Distilling the Knowledge in a Neural Network" by Hinton et al. The loss is scaled by the square of the temperature.
    Args:
        student_outputs (torch.Tensor): The output logits from the student model.
        teacher_outputs (dict): A dictionary containing the output logits from the teacher model, with the key 'pred_logits'.
        temprature (float): The temperature parameter used to soften the logits.
    Returns:
        torch.Tensor: The computed soft targets loss.
    """
    
    student_logits = student_outputs['pred_logits']
    teacher_logits = teacher_outputs['pred_logits']
    #Soften the student logits by applying softmax first and log() second
    soft_targets = torch.nn.functional.softmax(teacher_logits / temprature, dim=-1)
    soft_prob = torch.nn.functional.log_softmax(student_logits / temprature, dim=-1)
    # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
    soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temprature**2)
    return soft_targets_loss