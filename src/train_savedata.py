import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb  # 添加 wandb 导入

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def calculate_rectangle_vertices(width, height, x=0, y=0):
    """计算矩形顶点坐标"""
    return [
        {'x': x + width, 'y': y + height},
        {'x': x, 'y': y + height},
        {'x': x, 'y': y},
        {'x': x + width, 'y': y}
    ]


def construct_adam(arranged_parts, original_indices):
    """构建ADAM数据结构"""
    arranged_parts = arranged_parts.detach().cpu().numpy()[0]
    original_indices = original_indices.detach().cpu().numpy()
    adam = []

    for idx, (width, height) in enumerate(arranged_parts):
        if width > 0 and height > 0:
            adam.append([
                calculate_rectangle_vertices(width, height),
                {'id': int(original_indices[idx])},
                {'source': int(original_indices[idx])}
            ])

    return adam


class Attention(nn.Module):
    """自定义Attention层"""

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, values):
        # query: [batch_size, 1, hidden_dim]
        # values: [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        scores = torch.matmul(query, values.transpose(-2, -1))  # [batch_size, 1, seq_len]
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加权求和得到上下文向量
        context = torch.matmul(attention_weights, values)  # [batch_size, 1, hidden_dim]
        
        return context


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(PolicyNetwork, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention()
        self.dense = nn.Linear(hidden_dim, seq_len)
        self.rotation_output = nn.Linear(hidden_dim, 4)  # 4种旋转角度: 0, 90, 180, 270

        # 初始化权重
        nn.init.kaiming_normal_(self.rotation_output.weight)

    def encode_decode_step(self, inputs, decoder_input, decoder_state, selected_indices=None):
        """共用的编码-解码-注意力计算步骤"""
        # 编码
        encoder_output, _ = self.encoder(inputs)  # [batch_size, seq_len, hidden_dim]
        batch_size = inputs.size(0)

        # 如果没有decoder输入，创建一个零张量
        if decoder_input is None:
            decoder_input = torch.zeros((batch_size, 1, inputs.size(2)), device=inputs.device)

        # 如果没有decoder状态，初始化一个
        if decoder_state is None:
            h0 = torch.zeros((1, batch_size, self.hidden_dim), device=inputs.device)
            c0 = torch.zeros((1, batch_size, self.hidden_dim), device=inputs.device)
            decoder_state = (h0, c0)

        # 解码
        decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)  # [batch_size, 1, hidden_dim]

        # 注意力机制
        attention_output = self.attention(decoder_output, encoder_output)  # [batch_size, 1, hidden_dim]
        attention_output = attention_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # 生成logits
        logits = self.dense(attention_output)  # [batch_size, seq_len]

        # 处理mask - 将已选择的位置标记为负无穷
        if selected_indices is not None and len(selected_indices) > 0:
            # 确保mask和logits具有相同的数据类型
            mask = torch.zeros_like(logits, device=logits.device, dtype=logits.dtype)  # [batch_size, seq_len]
            for idx in selected_indices:
                # 确保idx_squeezed是2D的 [batch_size, 1]
                idx_squeezed = idx.squeeze(1).unsqueeze(1)  # [batch_size, 1]
                # 创建与mask相同数据类型的填充值
                ones = torch.ones_like(idx_squeezed, device=mask.device, dtype=mask.dtype)
                # 使用正确的维度进行scatter
                mask = mask.scatter_(1, idx_squeezed, ones)
            # 将已选择的位置标记为负无穷
            logits = torch.where(mask == 0, logits, torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))

        return encoder_output, logits, attention_output, decoder_state

    def forward(self, inputs, target_indices=None):
        """前向传播，支持训练和推理模式"""
        batch_size = inputs.size(0)
        decoder_input = None
        decoder_state = None

        # 使用Python列表进行收集
        all_logits = []
        all_rotation_logits = []
        selected_indices = []
        rotation_decisions = []

        for step in range(self.seq_len):
            # 编码-解码-注意力计算
            encoder_output, logits, attention_output, decoder_state = self.encode_decode_step(
                inputs, decoder_input, decoder_state,
                selected_indices if selected_indices else None
            )

            # 保存当前步骤的logits用于计算损失
            all_logits.append(logits)

            # 根据是否提供目标索引，选择预测方式
            if target_indices is not None and step < target_indices.size(1):
                # 训练模式：使用目标索引
                idx = target_indices[:, step:step + 1]
            else:
                # 推理模式：使用argmax
                idx = torch.argmax(logits, dim=1, keepdim=True)

            selected_indices.append(idx)

            # 旋转预测
            rotation_logits = self.rotation_output(attention_output)
            all_rotation_logits.append(rotation_logits)

            if target_indices is not None and step < target_indices.size(1):
                # 训练模式不需要做旋转决策，留给损失函数处理
                pass
            else:
                # 推理模式：选择最高概率的旋转
                rotation_decision = torch.argmax(rotation_logits, dim=1, keepdim=True)
                rotation_decisions.append(rotation_decision)

            # 更新decoder输入
            idx_squeezed = idx.squeeze(1)  # [batch_size]
            
            # 确保索引在有效范围内
            idx_squeezed = idx_squeezed.clamp(0, inputs.size(1) - 1)
            
            # 使用正确的索引方式
            batch_indices = torch.arange(batch_size, device=inputs.device)
            decoder_input = inputs[batch_indices, idx_squeezed].unsqueeze(1)  # [batch_size, 1, input_dim]

        # 合并所有结果
        all_logits = torch.stack(all_logits, dim=1)  # [batch_size, seq_len, seq_len]
        all_rotation_logits = torch.stack(all_rotation_logits, dim=1)  # [batch_size, seq_len, 4]
        selected_indices_tensor = torch.cat(selected_indices, dim=1)  # [batch_size, seq_len]

        if target_indices is None:
            rotation_decisions_tensor = torch.cat(rotation_decisions, dim=1)  # [batch_size, seq_len]
            return selected_indices_tensor, rotation_decisions_tensor, all_logits, all_rotation_logits
        else:
            return all_logits, all_rotation_logits

    def predict(self, inputs):
        """推理模式"""
        selected_indices, rotation_decisions, _, _ = self.forward(inputs)
        return selected_indices, rotation_decisions


class BinPackingDataset(Dataset):
    """用于装箱问题的数据集"""

    def __init__(self, jsonl_file):
        self.data = []
        self.load_jsonl_data(jsonl_file)

    def load_jsonl_data(self, jsonl_file):
        """加载JSONL格式的数据文件"""
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    item = json.loads(line)
                    self.data.append(item)
        logger.info(f"Loaded {len(self.data)} data samples from {jsonl_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 直接使用数据，不需要排序，因为已经排序好了
        bin_data = torch.tensor(item['bin'], dtype=torch.float32)
        parts = torch.tensor(item['parts'], dtype=torch.float32)
        placement_order = torch.tensor(item['placement_order'], dtype=torch.long)

        # 处理旋转角度 (0, 90, 180, 270) -> (0, 1, 2, 3)
        rotation_degrees = torch.tensor(item['rotation'], dtype=torch.float32)
        rotation_classes = (rotation_degrees / 90).long() % 4

        # 计算有效长度
        valid_length = len(item['parts'])

        return {
            'bin': bin_data,
            'parts': parts,
            'placement_order': placement_order,
            'rotation': rotation_classes,
            'efficiency': item['efficiency'],
            'valid_length': valid_length
        }


def collate_fn(batch):
    """处理不同长度的数据批次"""
    # 找出最大的parts长度
    max_len = max(len(item['parts']) for item in batch)
    
    # 初始化批次数据
    bins = []
    padded_parts = []
    padded_placement_orders = []
    padded_rotations = []
    efficiencies = []
    valid_lengths = []

    for item in batch:
        bins.append(item['bin'])

        # 对parts进行填充
        parts = item['parts']
        current_len = len(parts)
        padding_len = max_len - current_len
        padded = torch.cat([parts, torch.zeros((padding_len, 2), dtype=torch.float32)], dim=0)
        padded_parts.append(padded)

        # 对placement_order进行填充 - 确保长度为max_len
        placement_order = item['placement_order']
        # 检查并确保placement_order长度为current_len
        if len(placement_order) != current_len:
            # 如果长度不一致，截断或填充到current_len
            if len(placement_order) > current_len:
                placement_order = placement_order[:current_len]
            else:
                placement_order = torch.cat([
                    placement_order, 
                    torch.zeros(current_len - len(placement_order), dtype=torch.long)
                ])
        
        # 然后填充到max_len
        padded_order = torch.cat([placement_order, torch.zeros(padding_len, dtype=torch.long)])
        padded_placement_orders.append(padded_order)

        # 对rotation进行填充 - 同样确保长度一致
        rotation = item['rotation']
        # 检查并确保rotation长度为current_len
        if len(rotation) != current_len:
            # 如果长度不一致，截断或填充到current_len
            if len(rotation) > current_len:
                rotation = rotation[:current_len]
            else:
                rotation = torch.cat([
                    rotation, 
                    torch.zeros(current_len - len(rotation), dtype=torch.long)
                ])
        
        # 然后填充到max_len
        padded_rotation = torch.cat([rotation, torch.zeros(padding_len, dtype=torch.long)])
        padded_rotations.append(padded_rotation)

        efficiencies.append(item['efficiency'])
        valid_lengths.append(current_len)  # 使用实际长度而不是item['valid_length']

    # 打印调试信息
    lengths = [len(x) for x in padded_placement_orders]
    if len(set(lengths)) > 1:
        print(f"Warning: Unequal tensor lengths after padding: {lengths}")
    
    # 转换为张量
    bins = torch.stack(bins)
    padded_parts = torch.stack(padded_parts)
    padded_placement_orders = torch.stack(padded_placement_orders)
    padded_rotations = torch.stack(padded_rotations)
    valid_lengths = torch.tensor(valid_lengths, dtype=torch.long)

    return {
        'bins': bins,  # [batch_size, 2]
        'parts': padded_parts,  # [batch_size, max_len, 2]
        'placement_orders': padded_placement_orders,  # [batch_size, max_len]
        'rotations': padded_rotations,  # [batch_size, max_len]
        'efficiencies': torch.tensor(efficiencies, dtype=torch.float32),  # [batch_size]
        'valid_lengths': valid_lengths  # [batch_size]
    }


def train(model, train_loader, optimizer, device, epochs=10):
    """训练模型 - 修改为批量处理"""
    model.train()

    for epoch in range(epochs):
        total_placement_loss = 0
        total_rotation_loss = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            parts = batch['parts'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            rotation_targets = batch['rotations'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)

            # 重置梯度
            optimizer.zero_grad()

            # 前向传播
            placement_logits, rotation_logits = model(parts, placement_targets)

            # 计算损失 - 批量处理
            batch_size = parts.size(0)
            max_seq_len = parts.size(1)

            # 创建掩码，只对有效部分计算损失
            mask = torch.zeros_like(placement_targets, dtype=torch.bool)
            for i in range(batch_size):
                mask[i, :valid_lengths[i]] = True

            # 修正：在每个位置独立计算损失
            placement_loss = 0
            placement_count = 0
            rotation_loss = 0
            rotation_count = 0

            # 按批次单独处理每个样本
            for i in range(batch_size):
                # 只处理有效长度内的部分
                valid_len = valid_lengths[i].item()
                
                # 确保我们只计算有效长度内的损失
                if valid_len > 0:
                    # 计算放置顺序损失
                    for j in range(valid_len):
                        sample_loss = F.cross_entropy(
                            placement_logits[i, j].unsqueeze(0),  # [1, seq_len]
                            placement_targets[i, j].unsqueeze(0)  # [1]
                        )
                        placement_loss += sample_loss
                        placement_count += 1

                    # 计算旋转角度损失
                    for j in range(valid_len):
                        sample_loss = F.cross_entropy(
                            rotation_logits[i, j].unsqueeze(0),  # [1, 4]
                            rotation_targets[i, j].unsqueeze(0)  # [1]
                        )
                        rotation_loss += sample_loss
                        rotation_count += 1

            # 计算平均损失
            if placement_count > 0:
                placement_loss = placement_loss / placement_count
            
            if rotation_count > 0:
                rotation_loss = rotation_loss / rotation_count

            # 总损失
            total_loss = placement_loss + rotation_loss

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 更新统计信息
            valid_count = mask.sum().item()
            total_placement_loss += placement_loss.item() * valid_count
            total_rotation_loss += rotation_loss.item() * valid_count
            total_samples += valid_count

            # 记录批次级别的指标到 wandb
            wandb.log({
                'batch_placement_loss': placement_loss.item(),
                'batch_rotation_loss': rotation_loss.item(),
                'batch_total_loss': total_loss.item(),
                'batch': batch_idx + epoch * len(train_loader)
            })

            # 更新进度条
            progress_bar.set_postfix({
                'placement_loss': placement_loss.item(),
                'rotation_loss': rotation_loss.item()
            })

        # 计算并记录 epoch 级别的指标
        avg_placement_loss = total_placement_loss / total_samples if total_samples > 0 else 0
        avg_rotation_loss = total_rotation_loss / total_samples if total_samples > 0 else 0
        
        # 记录 epoch 级别的指标到 wandb
        wandb.log({
            'epoch_placement_loss': avg_placement_loss,
            'epoch_rotation_loss': avg_rotation_loss,
            'epoch_total_loss': avg_placement_loss + avg_rotation_loss,
            'epoch': epoch + 1
        })

        logger.info(f"Epoch {epoch + 1}/{epochs}: "
                    f"Placement Loss: {avg_placement_loss:.4f}, "
                    f"Rotation Loss: {avg_rotation_loss:.4f}")

    return model


def evaluate(model, val_loader, device):
    """评估模型 - 修改为批量处理"""
    model.eval()
    total_placement_loss = 0
    total_rotation_loss = 0
    correct_placements = 0
    correct_rotations = 0
    total_placements = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            # 移动数据到设备
            parts = batch['parts'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            rotation_targets = batch['rotations'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)

            # 前向传播
            placement_logits, rotation_logits = model(parts, placement_targets)

            # 创建掩码
            batch_size = parts.size(0)
            mask = torch.zeros_like(placement_targets, dtype=torch.bool)
            for i in range(batch_size):
                mask[i, :valid_lengths[i]] = True

            # 使用与train函数相同的逻辑计算损失
            placement_loss = 0
            placement_count = 0
            rotation_loss = 0
            rotation_count = 0

            # 按批次单独处理每个样本和位置
            for i in range(batch_size):
                valid_len = valid_lengths[i].item()
                
                if valid_len > 0:
                    # 计算放置顺序损失
                    for j in range(valid_len):
                        sample_loss = F.cross_entropy(
                            placement_logits[i, j].unsqueeze(0),
                            placement_targets[i, j].unsqueeze(0)
                        )
                        placement_loss += sample_loss
                        placement_count += 1

                    # 计算旋转角度损失
                    for j in range(valid_len):
                        sample_loss = F.cross_entropy(
                            rotation_logits[i, j].unsqueeze(0),
                            rotation_targets[i, j].unsqueeze(0)
                        )
                        rotation_loss += sample_loss
                        rotation_count += 1

            # 计算平均损失
            if placement_count > 0:
                placement_loss = placement_loss / placement_count
            
            if rotation_count > 0:
                rotation_loss = rotation_loss / rotation_count

            # 更新损失统计
            valid_count = mask.sum().item()
            total_placement_loss += placement_loss.item() * valid_count
            total_rotation_loss += rotation_loss.item() * valid_count

            # 计算准确率 - 逐个位置计算
            for i in range(batch_size):
                valid_len = valid_lengths[i].item()
                for j in range(valid_len):
                    pred_placement = torch.argmax(placement_logits[i, j])
                    if pred_placement == placement_targets[i, j]:
                        correct_placements += 1
                    
                    pred_rotation = torch.argmax(rotation_logits[i, j])
                    if pred_rotation == rotation_targets[i, j]:
                        correct_rotations += 1
                
                total_placements += valid_len

            # 记录批次级别的评估指标到 wandb
            wandb.log({
                'val_batch_placement_loss': placement_loss.item(),
                'val_batch_rotation_loss': rotation_loss.item(),
                'val_batch': batch_idx
            })

    # 计算平均指标
    avg_placement_loss = total_placement_loss / total_placements if total_placements > 0 else 0
    avg_rotation_loss = total_rotation_loss / total_placements if total_placements > 0 else 0
    placement_accuracy = correct_placements / total_placements if total_placements > 0 else 0
    rotation_accuracy = correct_rotations / total_placements if total_placements > 0 else 0

    # 记录最终的评估指标到 wandb
    wandb.log({
        'val_placement_loss': avg_placement_loss,
        'val_rotation_loss': avg_rotation_loss,
        'val_placement_accuracy': placement_accuracy,
        'val_rotation_accuracy': rotation_accuracy
    })

    logger.info(f"Evaluation: "
                f"Placement Loss: {avg_placement_loss:.4f}, "
                f"Rotation Loss: {avg_rotation_loss:.4f}, "
                f"Placement Accuracy: {placement_accuracy:.4f}, "
                f"Rotation Accuracy: {rotation_accuracy:.4f}")

    return {
        'placement_loss': avg_placement_loss,
        'rotation_loss': avg_rotation_loss,
        'placement_accuracy': placement_accuracy,
        'rotation_accuracy': rotation_accuracy
    }


def predict(model, test_data, device):
    """使用模型进行预测"""
    model.eval()
    results = []

    with torch.no_grad():
        for item in tqdm(test_data, desc="Predicting"):
            # 准备输入数据
            parts = torch.tensor(item['parts'], dtype=torch.float32).unsqueeze(0).to(device)

            # 预测放置顺序和旋转角度
            selected_indices, rotation_decisions = model.predict(parts)

            # 转换为Python列表
            placement_order = selected_indices[0].cpu().numpy().tolist()
            rotation_classes = rotation_decisions[0].cpu().numpy().tolist()

            # 将旋转类别转换为角度
            rotation_degrees = [int(cls * 90) % 360 for cls in rotation_classes]

            # 提取有效的预测（非零部分）
            valid_length = len(item['parts'])
            valid_placement = placement_order[:valid_length]
            valid_rotation = rotation_degrees[:valid_length]

            # 添加结果
            results.append({
                'bin': item['bin'],
                'parts': item['parts'],
                'predicted_placement': valid_placement,
                'predicted_rotation': valid_rotation
            })

    return results


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 初始化 wandb
    wandb.init(
        project="bin-packing",  # 项目名称
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "hidden_dim": 128,
            "input_dim": 2,
            "device": str(device)
        }
    )

    # 加载数据
    data_file = '../output/placement-0412.jsonl'
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return

    # 创建数据集
    dataset = BinPackingDataset(data_file)

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 增大批次大小以利用GPU并行性
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # 增大批次大小以利用GPU并行性
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    # 创建模型
    input_dim = 2  # width, height
    hidden_dim = 128

    # 获取最大序列长度
    max_seq_len = max(len(item['parts']) for i in range(len(dataset)) for item in [dataset[i]])
    logger.info(f"Maximum sequence length: {max_seq_len}")

    # 创建模型
    model = PolicyNetwork(input_dim, hidden_dim, max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    logger.info("Starting training...")
    model = train(model, train_loader, optimizer, device, epochs=10)

    # 评估模型
    logger.info("Evaluating model...")
    metrics = evaluate(model, val_loader, device)

    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/bin_packing_model.pth')
    logger.info("Model saved to models/bin_packing_model.pth")

    # 保存评估指标
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation metrics saved to models/evaluation_metrics.json")

    # 结束 wandb 运行
    wandb.finish()


if __name__ == '__main__':
    main()
