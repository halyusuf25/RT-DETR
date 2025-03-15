import torch

def matching_outputs_per_layer(matcher, batch_size, 
                               layer_id, 
                               student_outputs, 
                               teacher_outputs, 
                               topk_teacher_indices
):
    """
    Matches the outputs of the student model to the teacher model's outputs for a specific layer.
    Args:
        matcher (callable): A function or callable object that performs the matching between student and teacher outputs.
        batch_size (int): The number of samples in the batch.
        layer_id (int): The ID of the layer for which the matching is performed.
        student_outputs (dict): A dictionary containing the student model's outputs. 
            Expected keys are 'intermediate_logits' and 'intermediate_reference_points'.
        teacher_outputs (dict): A dictionary containing the teacher model's outputs.
            Expected keys are 'pred_logits_per_layer' and 'pred_boxes_per_layer'.
    Returns:
        dict: The result of the matcher function, which contains the matched outputs.
    """
    
    st_output = {"pred_logits": student_outputs['pred_logits_per_layer'][layer_id],
                 "pred_boxes": student_outputs['pred_boxes_per_layer'][layer_id]}
    targets = []
    for item in range(batch_size):
        _, class_labels = torch.max(teacher_outputs['pred_logits_per_layer'][layer_id, item, topk_teacher_indices[layer_id, item]], dim=-1)
        boxes = teacher_outputs['pred_boxes_per_layer'][layer_id, item, topk_teacher_indices[layer_id, item]]
        # targets.append({"class_labels": class_labels.detach().cpu(), "boxes": boxes.detach().cpu()})
        targets.append({"labels": class_labels, "boxes": boxes})

    matcher_outputs = matcher(st_output, targets)
    return matcher_outputs


def get_topk_queries(input_logits, k=100):
    """
    Get the top k queries from the input logits.
    Args:
        input_logits (torch.Tensor): The input logits tensor of shape [num_layers, batch_size, num_queries, num_classes].
        k (int): The number of top queries to get.
    Returns:
        topk_values (torch.Tensor): The top k values of the input logits tensor.
        topk_indices (torch.Tensor): The top k indices of the input logits tensor.
    """
    queries = torch.softmax(input_logits, dim=-1)   # Shape becomes [num_layers, batch_size, num_queries, num_classes]
    queries = queries.max(dim=-1, keepdim=True).values  # Shape becomes [num_layers, batch_size, num_queries, 1]
    queries = queries.squeeze(-1)  # Shape becomes [num_layers, batch_size, num_queries]
    topk_values, topk_indices = queries.topk(k=k, dim=-1)
    return topk_values, topk_indices

def get_topk_indices(teacher_topk_indices, matched_indices):   
    """
    Filters and matches the top-k indices from the teacher model with the matched indices.
    Args:
        teacher_topk_indices (torch.Tensor): A tensor containing the top-k indices from the teacher model.
        matched_indices (tuple of torch.Tensor): A tuple containing two tensors, where the first tensor represents 
                                                 the indices from the student model and the second tensor represents 
                                                 the corresponding matched indices from the teacher model.
    Returns:
        list: A list of matched indices from the student model.
        list: A list of matched top-k indices from the teacher model.
    """
   
    topk_indices = teacher_topk_indices.tolist()

    x_vals, y_vals = matched_indices  # Unpack the tuple of tensors
    
    only_topk_matched_indices = [(x.item(), y.item()) for x, y in zip(x_vals, y_vals) if y.item() in set(topk_indices)]
    
    st_topk_indices, tr_topk_indices = zip(*only_topk_matched_indices) if only_topk_matched_indices else ([], [])
    
    #match the topk_indices of the teacher
    tr_after_matching =  [topk_indices[i] for i in tr_topk_indices]

    return list(st_topk_indices), tr_after_matching


def compute_response_loss(student_outputs, teacher_outputs, temperature=1.0):
    """
    Computes the response distillation loss between the student and teacher model outputs.
    This function calculates the Kullback-Leibler divergence loss between the softmax outputs
    of the student and teacher models, which is used for knowledge distillation. The logits
    from both models are scaled by a temperature parameter before applying the softmax function.
    Args:
        student_outputs (dict): A dictionary containing the student model's outputs, 
                                expected to have a key 'pred_logits' with the logits tensor.
        teacher_outputs (dict): A dictionary containing the teacher model's outputs, 
                                expected to have a key 'pred_logits' with the logits tensor.
        temperature (float, optional): The temperature scaling factor for distillation. 
                                       Default is 1.0.
    Returns:
        torch.Tensor: The computed response distillation loss.
    """

    student_logits = student_outputs['pred_logits']
    teacher_logits = teacher_outputs['pred_logits']

    student_soft = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    response_distillation_loss = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
   
    return response_distillation_loss

def compute_attention_map_loss(student_outputs, 
                                teacher_outputs, 
                                student_model, 
                                teacher_model, 
                                matcher,
                                layers, 
                                teacher_topk_indices, 
                                attn_map_loss_fn=torch.nn.functional.mse_loss, 
):

    """
    Compute the attention map loss between the student and teacher models.
    Args:
        student_outputs (dict): The outputs from the student model, containing 'pred_logits'.
        teacher_outputs (dict): The outputs from the teacher model.
        student_model (torch.nn.Module): The student model.
        teacher_model (torch.nn.Module): The teacher model.
        matcher (callable): A function to match the outputs between the student and teacher models.
        layers (list): List of layer indices to compute the attention map loss for.
        teacher_topk_indices (torch.Tensor): Top-k indices from the teacher model for each layer and batch item.
        attn_map_loss_fn (callable, optional): The loss function to compute the attention map loss. Defaults to torch.nn.functional.mse_loss.
    Returns:
        float: The total attention map loss averaged over the batch size.
    """    
    batch_size = student_outputs['pred_logits'].shape[0]
    st_decoder_layers = student_model.module.decoder.decoder.layers  
    tr_decoder_layers = teacher_model.transformer.decoder.layers

    total_attn_map_loss=0
    for layer_id in layers:
        matched_indices = matching_outputs_per_layer(matcher, batch_size, layer_id, student_outputs, teacher_outputs, teacher_topk_indices)    
        loss_per_batch = 0
        for batch_item in range(batch_size):  
            st_quer_indices, tr_quer_indices = get_topk_indices(teacher_topk_indices[layer_id, batch_item], matched_indices[batch_item])           
            if not st_quer_indices or not tr_quer_indices:
                print(f"Warning: No matching indices found for batch_item {batch_item} in layer {layer_id}.")
                continue
            student_attn_weights = st_decoder_layers[layer_id].cross_attn_weights[batch_item, st_quer_indices]
            teacher_attn_weights = tr_decoder_layers[layer_id].cross_attn_weights[batch_item, tr_quer_indices]
            loss_per_batch += max(attn_map_loss_fn(student_attn_weights,teacher_attn_weights),0.0)
            # print(f"loss_per_batch: {loss_per_batch}")
        
        total_attn_map_loss += loss_per_batch / batch_size
        # print(f"total_attn_map_loss: {total_attn_map_loss}")
    return total_attn_map_loss


def compute_disttlation_loss(student_outputs, 
                             teacher_outputs, 
                             student_model, 
                             teacher_model, 
                             matcher, 
                             layers, 
                             teacher_topk_indices, 
                             attn_map_loss_fn=torch.nn.functional.mse_loss,
                             alpha=0.5,
):
    """
    Compute the total distillation loss between the student and teacher models.
    Args:
        student_outputs (dict): The outputs from the student model, containing 'pred_logits'.
        teacher_outputs (dict): The outputs from the teacher model.
        student_model (nn.Module): The student model.
        teacher_model (nn.Module): The teacher model.
        matcher (callable): A function to match the outputs of the student and teacher models.
        layers (list): List of layer indices to compute the loss on.
        teacher_topk_indices (list): List of top-k indices for the teacher model.
        attn_map_loss_fn (callable): A function to compute the attention map loss.
    Returns:
        float: The total distillation loss.
    """
    response_loss = compute_response_loss(student_outputs, teacher_outputs)
    attn_map_loss = compute_attention_map_loss(student_outputs, 
                                                teacher_outputs, 
                                                student_model, 
                                                teacher_model, 
                                                matcher, 
                                                layers, 
                                                teacher_topk_indices,
                                                attn_map_loss_fn=attn_map_loss_fn,)

    # attn_map_loss = torch.clamp(attn_map_loss, min=1e-3)
    # attn_map_loss = max(attn_map_loss, 1e-4)
    total_distillation_loss = (1-alpha) * response_loss + (alpha) * attn_map_loss
    return total_distillation_loss