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


def compute_attention_mapp_loss(student_outputs, teacher_outputs, student_model, teacher_model, matcher,
                                layers, scaling_tensor, student_topk_indices, teacher_topk_indices, 
                                batch_size, attn_map_loss_fn, h_, w_ , device):
    """
    Computes the attention map loss between the student and teacher models for specified layers.
    Args:
        student_outputs (dict): Outputs from the student model.
        teacher_outputs (dict): Outputs from the teacher model.
        student_model (nn.Module): The student model.
        teacher_model (nn.Module): The teacher model.
        layers (list): List of layer indices to compute the loss for.
        scaling_tensor (torch.Tensor): Tensor used for scaling coordinates.
        student_topk_indices (torch.Tensor): Top-k indices for the student model.
        teacher_topk_indices (torch.Tensor): Top-k indices for the teacher model.
        batch_size (int): The batch size.
        attn_map_loss_fn (callable): Loss function to compute the attention map loss.
        h_ (int): Height of the image.
        w_ (int): Width of the image.
        device (torch.device): Device to perform computations on.
    Returns:
        torch.Tensor: The total attention map loss.
    """
    
    
    
    st_decoder_layers = student_model.model.decoder.layers

    tr_decoder_layers = teacher_model.transformer.decoder.layers

    # mse_loss_fn = torch.nn.MSELoss()
    total_attn_map_loss=0
    for layer_id in layers:
        matched_indices = matching_outputs_per_layer(matcher, batch_size, layer_id, student_outputs, teacher_outputs, teacher_topk_indices)
        student_sampling_coords = st_decoder_layers[layer_id].cross_attn.sampling_locations
        teacher_sampling_coords = tr_decoder_layers[layer_id].cross_attn.sampling_locations

        st_attn_w = student_outputs['cross_attentions'][layer_id]

        tr_attn_w = tr_decoder_layers[layer_id].cross_attn.attention_weights_stored
        
        for batch_item in range(batch_size):  
            topk_indices = teacher_topk_indices[layer_id, batch_item].tolist()
            only_topk_matched_indices = [(x, y) for x, y in matched_indices[batch_item] if y in set(topk_indices)]
            st_topk_indices, tr_topk_indices = zip(*only_topk_matched_indices) if only_topk_matched_indices else ([], [])
            tr_after_matching =  [topk_indices[i] for i in tr_topk_indices]
          
            student_maps = generate_attention_maps_per_layer(batch_item, st_topk_indices, 
                                                             student_sampling_coords, 
                                                             st_attn_w, h_, w_, device)

            teacher_maps = generate_attention_maps_per_layer(batch_item, tr_after_matching, 
                                                             teacher_sampling_coords, 
                                                             tr_attn_w, h_, w_, device)
                        
            total_attn_map_loss += attn_map_loss_fn(teacher_maps.clone().detach().to(device), student_maps.clone().detach().to(device))
                
    return total_attn_map_loss