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

def get_topk_indices(layer_id, batch_item, teacher_topk_indices, matched_indices):   
    #find the topk-quiries indices for the teacher, and selecte the student queries accordingly
    print(f"matched_indices: {matched_indices}")
    topk_indices = teacher_topk_indices[layer_id, batch_item].tolist()
    print(f"topk_indices: {topk_indices}")
    x_vals, y_vals = matched_indices[batch_item]  # Unpack the tuple of tensors
    only_topk_matched_indices = [(x.item(), y.item()) for x, y in zip(x_vals, y_vals) if y.item() in set(topk_indices)]
    # only_topk_matched_indices = [(x, y) for x, y in matched_indices[batch_item] if y in set(topk_indices)]
    print(f"only_topk_matched_indices: {only_topk_matched_indices}")
    st_topk_indices, tr_topk_indices = zip(*only_topk_matched_indices) if only_topk_matched_indices else ([], [])

    #match the topk_indices of the teacher
    tr_after_matching =  [topk_indices[i] for i in tr_topk_indices]

    return st_topk_indices, tr_after_matching

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

def compute_attention_mapp_loss(student_outputs, 
                                teacher_outputs, 
                                student_model, 
                                teacher_model, 
                                matcher,
                                layers, 
                                teacher_topk_indices, 
                                attn_map_loss_fn, 
                                h_, w_ , 
                                device
):
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
    
    batch_size = student_outputs['pred_logits'].shape[0]
    # print(f"student model: {student_model}")
    st_decoder_layers = student_model.module.decoder.decoder.layers  
    tr_decoder_layers = teacher_model.transformer.decoder.layers

    total_attn_map_loss=0
    for layer_id in layers:
        matched_indices = matching_outputs_per_layer(matcher, batch_size, layer_id, student_outputs, teacher_outputs, teacher_topk_indices)
        
        st_attn_w = st_decoder_layers[layer_id].cross_attn.attention_weights_stored
        student_sampling_coords = st_decoder_layers[0].cross_attn.sampling_locations
        
        teacher_sampling_coords = tr_decoder_layers[0].cross_attn.sampling_locations
        tr_attn_w = tr_decoder_layers[layer_id].cross_attn.attention_weights_stored
        
        loss_per_batch = 0
        for batch_item in range(batch_size):  

            st_quer_indices, tr_quer_indices = get_topk_indices(layer_id, batch_item, teacher_topk_indices, matched_indices)

            student_attn_weights = st_decoder_layers[layer_id].cross_attn_weights[batch_item, st_quer_indices]
            teacher_attn_weights = tr_decoder_layers[layer_id].cross_attn_weights[batch_item, tr_quer_indices]
            # student_maps = generate_attention_maps_per_layer(batch_item, 
            #                                                  st_quer_indices, 
            #                                                  student_sampling_coords, 
            #                                                  st_attn_w, 
            #                                                  h_, w_, 
            #                                                  device
            # )

            # teacher_maps = generate_attention_maps_per_layer(batch_item, 
            #                                                  tr_quer_indices, 
            #                                                  teacher_sampling_coords, 
            #                                                  tr_attn_w, 
            #                                                  h_, w_, 
            #                                                  device
            # )
                        
            # loss_per_batch += attn_map_loss_fn(teacher_maps.clone().detach().to(device), student_maps.clone().detach().to(device))
            loss_per_batch += attn_map_loss_fn(teacher_attn_weights, student_attn_weights)
        
        total_attn_map_loss += loss_per_batch / batch_size

    return total_attn_map_loss / len(layers)
