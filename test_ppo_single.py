import torch
import numpy as np
import argparse
import os
import sys
import json
import logging
from typing import Dict, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.model.ppo_actor_critic import PPOActorCritic
from src.environment.td_environment import TDEnvironment, TDDataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, test_file: str, max_seq_len: int, num_instances: int = 100):
    """加载模型并在测试集上评估"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. 加载模型
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    logger.info(f"Loading model from {model_path}...")
    # PyTorch 2.6+ fix: weights_only=False needed for loading argparse.Namespace
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 获取模型参数 (尝试从 checkpoint 中获取，否则使用默认值)
    args = checkpoint.get('args')
    input_dim = getattr(args, 'input_dim', 2)
    hidden_dim = getattr(args, 'hidden_dim', 128)
    
    # 初始化模型结构
    policy = PPOActorCritic(input_dim, hidden_dim, max_seq_len).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    logger.info("Model loaded successfully.")

    # 2. 加载数据和环境
    if not os.path.exists(test_file):
        logger.error(f"Test data file not found: {test_file}")
        return

    test_loader = TDDataLoader(test_file, test_file) # 这里为了简单，train_file 也传 test_file
    # 注意：TDDataLoader 的第二个参数才是 test_file，但它内部逻辑有点混淆，这里我们只用 get_test_instance
    # 实际上我们只需要 test_instances
    
    # 如果 test_loader 没有加载到数据，手动尝试加载
    if not test_loader.test_instances:
         logger.info(f"Loading test instances directly from {test_file}")
         with open(test_file, 'r') as f:
            for line in f:
                test_loader.test_instances.append(json.loads(line))

    env = TDEnvironment(max_parts=max_seq_len)
    
    num_instances = min(num_instances, len(test_loader.test_instances))
    logger.info(f"Evaluating on {num_instances} instances...")

    total_util = 0.0
    results = []

    for i in range(num_instances):
        instance = test_loader.get_test_instance(i)
        if instance is None: 
            # 兼容处理，有时 loader 放在 test_instances 列表里
            instance = test_loader.test_instances[i]

        state_np = env.reset(instance)
        
        # 记录开始时间
        start_time = time.time()
        
        with torch.no_grad():
            state = torch.tensor(state_np, dtype=torch.float32, device=device)
            
            # 编码
            encoder_outputs = policy.encode_inputs(state.unsqueeze(0)).squeeze(0)
            
            # 解码
            decoder_input, decoder_hidden = policy.get_initial_state(1, device)
            decoder_input = decoder_input.squeeze(0)
            selected_mask = torch.zeros(max_seq_len, dtype=torch.bool, device=device)
            valid_length = torch.tensor([env.num_parts], device=device)

            done = False
            while not done:
                logits, _, next_decoder_hidden = policy.forward(
                    encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                    decoder_hidden, selected_mask.unsqueeze(0), valid_length
                )
                # 贪心选择
                action = torch.argmax(logits, dim=1)
                
                _, _, done, _ = env.step(action.item())
                
                decoder_hidden = next_decoder_hidden
                decoder_input = encoder_outputs[action.item()]
                selected_mask[action.item()] = True
        
        metrics = env.get_placement_metrics()
        util = metrics.get('material_utilization', 0)
        total_util += util
        
        elapsed = time.time() - start_time
        logger.info(f"Instance {i}: Util={util:.4f}, Time={elapsed:.2f}s")
        
        results.append({
            'instance_id': i,
            'utilization': util,
            'metrics': metrics
        })

    avg_util = total_util / num_instances
    logger.info(f"\n{'='*30}")
    logger.info(f"Final Average Utilization: {avg_util:.4f}")
    logger.info(f"{'='*30}")

if __name__ == '__main__':
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_ppo/ppo_policy_best.pth')
    parser.add_argument('--test_file', type=str, default='./data/placement-0529-ga-20epoch-norotation/test.jsonl')
    parser.add_argument('--max_seq_len', type=int, default=60)
    parser.add_argument('--num_instances', type=int, default=50)
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_file, args.max_seq_len, args.num_instances)

