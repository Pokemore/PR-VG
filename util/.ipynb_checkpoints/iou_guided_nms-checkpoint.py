"""
IoU引导的NMS后处理模块
"""
import torch
import torch.nn.functional as F
from util import box_ops

def iou_guided_nms(pred_boxes, pred_scores, pred_iou, iou_threshold=0.5, score_threshold=0.1):
    """
    IoU引导的NMS - 简化版本
    
    Args:
        pred_boxes: [N, 4] 预测边界框 (cxcywh格式)
        pred_scores: [N] 分类分数
        pred_iou: [N] 预测的IoU值
        iou_threshold: IoU阈值
        score_threshold: 分数阈值
    
    Returns:
        keep_indices: 保留的框的索引
    """
    valid_mask = pred_scores > score_threshold
    if not valid_mask.any():
        return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
    
    valid_boxes = pred_boxes[valid_mask]
    valid_scores = pred_scores[valid_mask]
    valid_iou = pred_iou[valid_mask]
    valid_indices = torch.where(valid_mask)[0]
    
    combined_scores = valid_scores * valid_iou
    
    best_idx = combined_scores.argmax()
    return torch.tensor([valid_indices[best_idx]], dtype=torch.long, device=pred_boxes.device)

def multi_query_iou_guided_selection(pred_logits, pred_boxes, pred_iou, num_queries=10):
    """
    多query的IoU引导选择
    
    Args:
        pred_logits: [batch_size, time, num_queries, num_classes] 或 [batch_size, num_queries, num_classes]
        pred_boxes: [batch_size, time, num_queries, 4] 或 [batch_size, num_queries, 4]
        pred_iou: [batch_size, time, num_queries, 1] 或 [batch_size, num_queries, 1]
        num_queries: query数量
    
    Returns:
        selected_indices: 每个样本选择的query索引
        selected_boxes: 选择的边界框
        selected_scores: 选择的分数
        selected_iou: 选择的IoU预测
    """
    if len(pred_logits.shape) == 4:
        batch_size, time, num_queries, num_classes = pred_logits.shape
        has_time_dim = True
    elif len(pred_logits.shape) == 3:
        batch_size, num_queries, num_classes = pred_logits.shape
        time = 1
        has_time_dim = False
        pred_logits = pred_logits.unsqueeze(1)
        pred_boxes = pred_boxes.unsqueeze(1)
        pred_iou = pred_iou.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported pred_logits shape: {pred_logits.shape}")
    
    selected_indices = []
    selected_boxes = []
    selected_scores = []
    selected_iou = []
    
    for b in range(batch_size):
        for t in range(time):
            logits = pred_logits[b, t]
            boxes = pred_boxes[b, t]
            iou_pred = pred_iou[b, t].squeeze(-1)
            
            scores = logits.sigmoid()
            max_scores, _ = scores.max(-1)
            
            keep_indices = iou_guided_nms(boxes, max_scores, iou_pred)
            
            if len(keep_indices) > 0:
                best_idx = keep_indices[0]
                selected_indices.append(best_idx)
                selected_boxes.append(boxes[best_idx])
                selected_scores.append(max_scores[best_idx])
                selected_iou.append(iou_pred[best_idx])
            else:
                best_idx = max_scores.argmax()
                selected_indices.append(best_idx)
                selected_boxes.append(boxes[best_idx])
                selected_scores.append(max_scores[best_idx])
                selected_iou.append(iou_pred[best_idx])
    
    return (torch.stack(selected_indices), 
            torch.stack(selected_boxes), 
            torch.stack(selected_scores), 
            torch.stack(selected_iou))

def adaptive_iou_threshold_selection(pred_boxes, pred_scores, pred_iou, 
                                   iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    自适应IoU阈值选择
    
    Args:
        pred_boxes: [N, 4] 预测边界框
        pred_scores: [N] 分类分数
        pred_iou: [N] 预测的IoU值
        iou_thresholds: 候选IoU阈值列表
    
    Returns:
        best_boxes: 最佳边界框
        best_scores: 最佳分数
        best_iou: 最佳IoU预测
        used_threshold: 使用的阈值
    """
    best_boxes = None
    best_scores = None
    best_iou = None
    best_threshold = None
    best_combined_score = -1
    
    for threshold in iou_thresholds:
        keep_indices = iou_guided_nms(pred_boxes, pred_scores, pred_iou, threshold)
        
        if len(keep_indices) > 0:
            combined_scores = pred_scores[keep_indices] * pred_iou[keep_indices]
            max_combined_score = combined_scores.max()
            
            if max_combined_score > best_combined_score:
                best_combined_score = max_combined_score
                best_idx = keep_indices[combined_scores.argmax()]
                best_boxes = pred_boxes[best_idx]
                best_scores = pred_scores[best_idx]
                best_iou = pred_iou[best_idx]
                best_threshold = threshold
    
    if best_boxes is None:
        best_idx = pred_scores.argmax()
        best_boxes = pred_boxes[best_idx]
        best_scores = pred_scores[best_idx]
        best_iou = pred_iou[best_idx]
        best_threshold = "fallback"
    
    return best_boxes, best_scores, best_iou, best_threshold
