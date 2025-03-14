import torch

def matching_outputs_per_layer(matcher, batch_size, layer_id, student_outputs, teacher_outputs, topk_teacher_indices):
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

def generate_attention_maps_per_layer(
        batch_index, topk_queries, sampled_coords, 
        attention_weights, img_h, img_w, device
):

    """
    Generate attention maps for each layer in a batch.
    Args:
        batch_index (int): Index of the batch to process.
        num_queries (int): Number of queries.
        sampled_coords (torch.Tensor): Coordinates sampled from the feature map, 
            shape (batch_size, num_queries, num_heads, num_feature_level, 2).
        attention_weights (torch.Tensor): Attention weights, 
            shape (batch_size, num_queries, num_heads, num_feature_level, num_points).
        topk_quries (list[int]): List of top-k query indices.
        img_h (int): Height of the image.
        img_w (int): Width of the image.
        device (torch.device): Device to perform computations on.
    Returns:
        torch.Tensor: Normalized attention maps, shape (num_queries, img_h, img_w).
    """
    
    _, num_queries, num_heads, num_feature_level, num_points = attention_weights.shape

    num_queries = len(topk_queries)
    num_feature_level = int(3)
    # Flatten all queries/heads/levels/points into one dimension
    coords_flat = sampled_coords[batch_index, topk_queries, : , :num_feature_level].reshape(-1, 2)
    rows, cols = coords_flat.unbind(dim=1)

    # Initialize the attention map
    query_map = torch.zeros((num_queries, num_heads, num_feature_level, img_h, img_w), dtype=torch.float32, device=device)
    
    # Expand the indices to match the dimensions of query_map
    query_idx = torch.arange(num_queries, dtype=torch.long).view(-1, 1, 1, 1, 1).expand(num_queries, num_heads, num_feature_level, num_points, 1).reshape(-1)
    head_idx = torch.arange(num_heads, dtype=torch.long).view(1, -1, 1, 1, 1).expand(num_queries, num_heads, num_feature_level, num_points, 1).reshape(-1)
    level_idx = torch.arange(num_feature_level, dtype=torch.long).view(1, 1, -1, 1, 1).expand(num_queries, num_heads, num_feature_level, num_points, 1).reshape(-1)
    
    # update query_map with the expanded indices
    query_map.index_put_((query_idx, head_idx, level_idx, rows, cols), attention_weights[batch_index, topk_queries, : , :num_feature_level].reshape(-1), accumulate=True)
        
    # final_maps = torch.mean(torch.sum(query_map, dim=2), dim=1)
    final_maps = query_map.sum(dim=(1,2))
    return final_maps / (num_heads * num_feature_level)


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

# def get_topk_indices(layer_id, batch_item, teacher_topk_indices, matched_indices):   
#     #find the topk-quiries indices for the teacher, and selecte the student queries accordingly
#     print(f"below print statements are for troubleshooting get_topk_indices() function")
#     print(f"layer_id: {layer_id}, batch_item: {batch_item}")
#     print(f"matched_indices: {matched_indices}")
#     topk_indices = teacher_topk_indices[layer_id, batch_item].tolist()
#     print(f"topk_indices: {topk_indices}")
#     x_vals, y_vals = matched_indices[batch_item]  # Unpack the tuple of tensors
#     print(f"x_vals: {x_vals}")
#     print(f"y_vals: {y_vals}")
#     only_topk_matched_indices = [(x.item(), y.item()) for x, y in zip(x_vals, y_vals) if y.item() in set(topk_indices)]
#     print(f"only_topk_matched_indices: {only_topk_matched_indices}")
#     st_topk_indices, tr_topk_indices = zip(*only_topk_matched_indices) if only_topk_matched_indices else ([], [])
#     print(f"st_topk_indices: {st_topk_indices}")
#     print(f"tr_topk_indices: {tr_topk_indices}")
#     #match the topk_indices of the teacher
#     tr_after_matching =  [topk_indices[i] for i in tr_topk_indices]
#     print(f"tr_after_matching: {tr_after_matching}")

#     return st_topk_indices, tr_after_matching

def get_topk_indices(layer_id, batch_item, teacher_topk_indices, matched_indices):   
    print(f"--- Debugging get_topk_indices() ---")
    print(f"layer_id: {layer_id}, batch_item: {batch_item}")
    
    topk_indices = teacher_topk_indices[layer_id, batch_item].tolist()
    
    x_vals, y_vals = matched_indices[batch_item]  # Unpack the tuple of tensors
    
    only_topk_matched_indices = [(x.item(), y.item()) for x, y in zip(x_vals, y_vals) if y.item() in set(topk_indices)]
    
    if not only_topk_matched_indices:
        print(f"Warning: No matching indices found for batch_item {batch_item}.")
    
    print(f"only_topk_matched_indices: {only_topk_matched_indices}")
    
    st_topk_indices, tr_topk_indices = zip(*only_topk_matched_indices) if only_topk_matched_indices else ([], [])
    print(f"st_topk_indices: {st_topk_indices}")
    print(f"tr_topk_indices: {tr_topk_indices}")

    # Check if all tr_topk_indices exist within range of topk_indices
    valid_tr_topk_indices = [i for i in tr_topk_indices if i < len(topk_indices)]
    
    if len(valid_tr_topk_indices) != len(tr_topk_indices):
        print(f"Warning: Some indices in tr_topk_indices are out of range! Invalid: {set(tr_topk_indices) - set(valid_tr_topk_indices)}")

    # Match the top-k indices of the teacher while ensuring indices are within range
    tr_after_matching = [topk_indices[i] for i in valid_tr_topk_indices]
    
    print(f"tr_after_matching: {tr_after_matching}")
    print(f"len(st_topk_indices): {len(st_topk_indices)}")
    print(f"len(tr_after_matching): {len(tr_after_matching)}")

    return st_topk_indices, tr_after_matching


def compute_response_loss(student_outputs, teacher_outputs, temperature=1.0):
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
        student_model (nn.Module): The student model.
        teacher_model (nn.Module): The teacher model.
        matcher (callable): A function to match the outputs of the student and teacher models.
        layers (list): List of layer indices to compute the loss on.
        teacher_topk_indices (list): List of top-k indices for the teacher model.
        attn_map_loss_fn (callable): A function to compute the attention map loss.
    Returns:
        float: The total attention map loss averaged over the batch.
    """
    
    batch_size = student_outputs['pred_logits'].shape[0]
    print(f"batch_size: {batch_size}")
    st_decoder_layers = student_model.module.decoder.decoder.layers  
    tr_decoder_layers = teacher_model.transformer.decoder.layers

    total_attn_map_loss=0
    for layer_id in layers:
        matched_indices = matching_outputs_per_layer(matcher, batch_size, layer_id, student_outputs, teacher_outputs, teacher_topk_indices)    
        loss_per_batch = 0
        for batch_item in range(batch_size):  
            st_quer_indices, tr_quer_indices = get_topk_indices(layer_id, batch_item, teacher_topk_indices, matched_indices)           
            print(f"len(st_quer_indices): {len(st_quer_indices)}")
            print(f"len(tr_quer_indices): {len(tr_quer_indices)}")
            student_attn_weights = st_decoder_layers[layer_id].cross_attn_weights[batch_item, st_quer_indices]
            teacher_attn_weights = tr_decoder_layers[layer_id].cross_attn_weights[batch_item, tr_quer_indices]
            # print(f"layer#: {layer_id}, batch_item#: {batch_item}")
            # print(f"student_attn_weights: {student_attn_weights}")
            # print(f"teacher_attn_weights: {teacher_attn_weights}")
            if torch.isnan(teacher_attn_weights).any() or torch.isnan(student_attn_weights).any():
                print("NaNs in teacher/student attention weights before MSE.")

            loss_per_batch += max(attn_map_loss_fn(student_attn_weights,teacher_attn_weights),0.0)
            print(f"loss_per_batch: {loss_per_batch}")
        

        total_attn_map_loss += loss_per_batch / batch_size
        print(f"total_attn_map_loss: {total_attn_map_loss}")

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