import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import glob
import os
import json
import datetime
import logging
import sys
from model import PolicyNetwork, calculate_rectangle_vertices, construct_adam

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# FastAPI应用初始化
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局状态管理
class GlobalState:
    def __init__(self):
        self.S_opt = {'placement': [], 'rotation': []}
        self.best_reward = float('-inf')
        self.bin_data = {}
        self.height_value = 0
        self.current_svg_content = None
        self.batch_num = 0
        self.is_training_started = False
        self.can_get_sopt = False
        self.can_get_svg = False
        self.end_epoch = 1000

        # 事件
        self.sopt_event = threading.Event()
        self.height_event = threading.Event()
        self.svg_event = threading.Event()


state = GlobalState()


# 数据处理函数
def load_data(file_pattern):
    """加载数据文件"""
    files = glob.glob(file_pattern)
    all_data, all_bin, all_indices, all_filenames = [], [], [], []

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            bin_data = np.array([float(x) for x in lines[0].split()[:2]], dtype=np.float32)
            items = np.zeros((len(lines) - 1, 2), dtype=np.float32)
            indices = np.arange(len(lines) - 1)

            for idx, line in enumerate(lines[1:]):
                width, height = map(float, line.split()[:2])
                items[idx] = [width, height]

            if len(items) > 0:
                areas = items[:, 0] * items[:, 1]
                sort_idx = np.argsort(-areas)  # 按面积从大到小排序
                items = items[sort_idx]
                indices = indices[sort_idx]

                all_data.append(items)
                all_bin.append(bin_data)
                all_indices.append(indices)
                file_name = str(os.path.splitext(os.path.basename(file))[0])
                all_filenames.append(file_name)

    return all_data, all_bin, all_indices, all_filenames


def get_svg_content(file_name):
    """读取SVG文件内容"""
    svg_path = os.path.join('../output_svg', f'{str(file_name)}.svg')
    try:
        with open(svg_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading SVG file {svg_path}: {e}")
        return None


def prepare_training(batch_file_names):
    """准备训练数据"""
    file_name = batch_file_names[0].numpy().decode()
    svg_content = get_svg_content(file_name)
    state.current_svg_content = svg_content
    state.svg_event.set()


# 训练相关函数
@tf.function
def train_step(model, optimizer, inputs, bins, original_indices):
    """单步训练"""
    # todo: add bins as parameter
    with tf.GradientTape() as tape:
        selected_indices, log_probs, rotation_decisions, rotation_log_probs = model(inputs)
        batch_size = tf.shape(inputs)[0]
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1),
                                [1, tf.shape(selected_indices)[1]])
        selected_indices = tf.cast(selected_indices, tf.int32)
        gather_indices = tf.stack([batch_indices, selected_indices], axis=2)
        arranged_inputs = tf.gather_nd(inputs, gather_indices)

        reordered_original_indices = tf.gather(original_indices[0], selected_indices[0])

        reward = tf.py_function(
            func=compute_packing_area,
            inp=[arranged_inputs, reordered_original_indices, rotation_decisions],
            Tout=tf.float32
        )

        position_loss = tf.reduce_mean(-log_probs * reward)
        rotation_loss = tf.reduce_mean(-rotation_log_probs * reward)
        total_loss = position_loss + rotation_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, reward


def compute_packing_area(arranged_parts, original_indices, rotation_decisions):
    """计算打包面积"""
    height = None
    num_attempts = 0

    while height is None and num_attempts < 10:
        num_attempts += 1
        adam = construct_adam(arranged_parts, original_indices)
        arranged_parts_np = arranged_parts.numpy()[0]
        rotations = []
        rotation_decisions_np = rotation_decisions.numpy()[0]

        for idx, (width, height) in enumerate(arranged_parts_np):
            if width > 0 and height > 0:
                rotations.append(90 if rotation_decisions_np[idx] == 1 else 0)

        state.S_opt['placement'] = adam
        state.S_opt['rotation'] = rotations
        state.sopt_event.set()

        height = send_sopt_and_receive_height()
        logger.info(f"Received height: {height}")

    if height is None:
        logger.warning("Failed to get valid height after 10 attempts")
        return 0

    return height


# FastAPI路由
class HeightData(BaseModel):
    height: float


@app.get('/getsvg')
def get_svg():
    """获取SVG内容"""
    timeout = 10
    if not state.can_get_svg:
        return {'status': 'not_ready', 'message': 'getsvg is not allowed'}

    if state.svg_event.wait(timeout):
        state.svg_event.clear()
        state.can_get_sopt = True
        state.can_get_svg = False
        return {
            'status': 'success',
            'svg_content': state.current_svg_content,
            'batch_num': state.batch_num
        }
    else:
        return {'status': 'not_ready', 'message': f'Timeout after {timeout} seconds'}


@app.get('/train')
def start_train():
    """开始训练"""
    try:
        if not state.is_training_started:
            logger.info("Starting training!")
            state.can_get_svg = True
            state.is_training_started = True
            threading.Thread(target=start_training).start()
            return {'status': 'success'}
        return {'status': 'already_running'}
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@app.get('/predict')
def start_predict():
    """开始推理"""
    try:
        if not state.is_training_started:
            if not os.path.exists('final_model'):
                return {
                    'status': 'error',
                    'message': 'No trained model found. Please train the model first.'
                }

            logger.info("Starting predicting!")
            state.can_get_svg = True
            threading.Thread(target=start_prediction).start()
            return {'status': 'success'}
        return {'status': 'already_running'}
    except Exception as e:
        logger.error(f"Error starting predicting: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@app.get('/getsopt')
def send_sopt():
    """获取优化布局结果"""
    timeout = 10
    if not state.can_get_sopt:
        logger.debug("getsopt is not allowed")
        return {
            'status': 'not_ready',
            'message': 'Must successfully get SVG first'
        }
    try:
        if state.sopt_event.wait(timeout):
            logger.info(f"Sending S_opt: {state.S_opt}, bin_data: {state.bin_data}, batch_num: {state.batch_num}")
            state.sopt_event.clear()
            state.can_get_sopt = False
            return {
                'status': 'Success',
                'S_opt': state.S_opt,
                'bin': state.bin_data,
                'batch_num': state.batch_num
            }
        else:
            logger.warning(f'Timeout after {timeout} seconds waiting for Sopt')
            return {
                'status': 'not_ready',
                'message': f'Timeout after {timeout} seconds waiting for Sopt'
            }
    except Exception as e:
        logger.error(f'Error in send_sopt: {e}')
        return {'status': 'error', 'message': str(e)}


@app.post('/sendheight')
def receive_height(data: HeightData):
    """接收高度数据"""
    state.height_value = data.height
    state.height_event.set()
    state.can_get_svg = True
    return {'status': 'success'}


def send_sopt_and_receive_height():
    """发送优化布局并等待高度响应"""
    state.height_event.wait()
    state.height_event.clear()
    return state.height_value


def init_processing():
    """初始化处理状态"""
    state.current_svg_content = None
    state.can_get_sopt = False
    state.can_get_svg = True


def save_batch_metrics(metrics_dir, epoch, batch_metrics):
    """
    保存每个batch的指标数据
    Args:
        metrics_dir: 保存目录
        epoch: 当前epoch
        batch_metrics: 包含batch指标的列表
    """
    os.makedirs(metrics_dir, exist_ok=True)
    filename = os.path.join(metrics_dir, f'epoch_{epoch}_batch_metrics.json')

    with open(filename, 'w') as f:
        json.dump(batch_metrics, f, indent=2)
    logger.info(f"Saved batch metrics for epoch {epoch}")


def train_model(resume_from=None):
    """完整的模型训练函数"""
    tf.config.run_functions_eagerly(True)

    # 加载数据
    data, bins, indices, file_names = load_data('../train/*.txt')
    max_seq_len = max(len(x) for x in data)

    # 数据预处理
    padded_data = np.array([
        np.pad(x, ((0, max_seq_len - len(x)), (0, 0)),
               mode='constant',
               constant_values=0)
        for x in data
    ])

    padded_indices = np.array([
        np.pad(x, (0, max_seq_len - len(x)),
               mode='constant',
               constant_values=-1)
        for x in indices
    ])

    # 准备训练数据
    combined_data = list(zip(padded_data, bins, padded_indices, file_names))
    train_combined, test_combined = train_test_split(combined_data, test_size=0.9, random_state=42)

    train_inputs = np.array([x[0] for x in train_combined])
    train_bins = np.array([x[1] for x in train_combined])
    train_indices = [x[2] for x in train_combined]
    train_file_names = [x[3] for x in train_combined]

    # 模型初始化
    input_dim = 2
    hidden_dim = 128
    batch_size = 1
    model = PolicyNetwork(input_dim, hidden_dim, max_seq_len)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    # 检查点管理
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_dir = '../training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # 指标保存目录
    metrics_dir = '../training_metrics'
    os.makedirs(metrics_dir, exist_ok=True)

    # 恢复训练状态
    start_epoch = 0
    start_batch = 0
    total_batches = 0
    total_loss = 0
    total_reward = 0

    if resume_from:
        checkpoint.restore(resume_from)
        with open(os.path.join(checkpoint_dir, 'training_state.jsonl'), 'r') as f:
            lines = f.readlines()
            training_state = json.loads(lines[-1].strip())
            start_epoch = training_state['epoch']
            start_batch = training_state['batch']
            total_batches = training_state.get('total_batches', 0)
            total_loss = training_state['avg_loss'] * total_batches
            total_reward = training_state['avg_reward'] * total_batches
        logger.info(f"Resuming from epoch {start_epoch}, batch {start_batch}")

    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((
        train_inputs,
        train_bins,
        train_indices,
        train_file_names
    )).batch(batch_size)

    # 用于存储所有epoch的metrics
    all_epochs_metrics = {}

    # 训练循环
    for epoch in tqdm(range(start_epoch, state.end_epoch)):
        batch_num = 0
        epoch_batch_metrics = []  # 存储当前epoch的所有batch数据
        current_chunk_metrics = []  # 存储当前100个batch的数据

        for batch_inputs, batch_bins, batch_indices, batch_file_names in dataset:
            state.batch_num = batch_num

            if epoch == start_epoch and batch_num < start_batch:
                batch_num += 1
                continue

            if batch_num == 0:
                total_batches = 0
                total_loss = 0
                total_reward = 0

            # 准备训练数据
            prepare_training(batch_file_names)
            state.bin_data = [[
                calculate_rectangle_vertices(
                    float(batch_bins[0][0]),
                    float(batch_bins[0][1])
                ),
                {'id': -1.0},
                {'source': -1.0}
            ]]

            # 训练步骤
            loss, reward = train_step(model, optimizer, batch_inputs, batch_bins, batch_indices)
            total_loss += loss
            total_reward += reward
            total_batches += 1

            # 记录当前batch的metrics
            batch_metric = {
                'batch_num': batch_num,
                'loss': float(loss),
                'reward': float(reward),
                'timestamp': datetime.datetime.now().isoformat()
            }
            epoch_batch_metrics.append(batch_metric)
            current_chunk_metrics.append(batch_metric)

            logger.info(f"Epoch {epoch}, Batch {batch_num}, Loss: {loss:.4f}, Reward: {reward:.4f}")

            # 每100个batch保存一次数据
            if len(current_chunk_metrics) == 100:
                logger.info(f"Saving metrics for epoch {epoch} and batches {batch_num-99} to {batch_num}")
                # chunk_filename = os.path.join(
                #     metrics_dir,
                #     f'epoch_{epoch}_batches_{batch_num - 99}_to_{batch_num}.json'
                # )
                # with open(chunk_filename, 'w') as f:
                #     json.dump(current_chunk_metrics, f, indent=2)
                current_chunk_metrics = []  # 重置当前chunk

                # 保存检查点
                checkpoint.save(file_prefix=checkpoint_prefix)
                logger.info(f"Checkpoint saved at {checkpoint_prefix}")
                training_state = {
                    'epoch': epoch,
                    'batch': batch_num + 1,
                    'total_batches': total_batches,
                    'avg_loss': float(total_loss / total_batches),
                    'avg_reward': float(total_reward / total_batches),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                with open(os.path.join(checkpoint_dir, 'training_state.jsonl'), 'a') as f:
                    json.dump(training_state, f)
                    f.write('\n')
                logger.info(f"Checkpoint and metrics saved at total batch {total_batches}")

            batch_num += 1

        # 保存最后不足100个batch的数据
        if current_chunk_metrics:
            chunk_filename = os.path.join(
                metrics_dir,
                f'epoch_{epoch}_batches_{batch_num - len(current_chunk_metrics)}_to_{batch_num - 1}.json'
            )
            with open(chunk_filename, 'w') as f:
                json.dump(current_chunk_metrics, f, indent=2)

        # 保存整个epoch的数据
        all_epochs_metrics[f'epoch_{epoch}'] = {
            'batch_metrics': epoch_batch_metrics,
            'epoch_summary': {
                'avg_loss': float(total_loss / total_batches),
                'avg_reward': float(total_reward / total_batches),
                'total_batches': total_batches
            }
        }

        # 输出epoch统计
        avg_loss = total_loss / total_batches
        avg_reward = total_reward / total_batches
        logger.info(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

    # 保存所有epoch的完整数据
    with open(os.path.join(metrics_dir, 'all_epochs_metrics.json'), 'w') as f:
        json.dump(all_epochs_metrics, f, indent=2)
    logger.info("Saved complete training metrics")

    # 保存最终模型
    tf.saved_model.save(model, 'final_model')
    logger.info("Final model saved")


def predict_model():
    """模型预测函数"""
    tf.config.run_functions_eagerly(True)

    # 加载数据
    data, bins, indices, file_names = load_data('../train/*.txt')
    max_seq_len = max(len(x) for x in data)

    # 数据预处理
    padded_data = np.array([
        np.pad(x, ((0, max_seq_len - len(x)), (0, 0)),
               mode='constant',
               constant_values=0)
        for x in data
    ])
    padded_indices = np.array([
        np.pad(x, (0, max_seq_len - len(x)),
               mode='constant',
               constant_values=-1)
        for x in indices
    ])

    # 准备测试数据
    combined_data = list(zip(padded_data, bins, padded_indices, file_names))
    _, test_combined = train_test_split(combined_data, test_size=0.1, random_state=42)

    test_inputs = np.array([x[0] for x in test_combined])
    test_bins = np.array([x[1] for x in test_combined])
    test_indices = [x[2] for x in test_combined]
    test_file_names = [x[3] for x in test_combined]

    dataset = tf.data.Dataset.from_tensor_slices((
        test_inputs,
        test_bins,
        test_indices,
        test_file_names
    )).batch(1)

    # 加载训练好的模型
    try:
        model = tf.saved_model.load('final_model')
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # 预测循环
    results = []
    for batch_num, (batch_inputs, batch_bins, batch_indices, batch_file_names) in enumerate(dataset):
        state.batch_num = batch_num
        prepare_training(batch_file_names)
        state.bin_data = [[
            calculate_rectangle_vertices(
                float(batch_bins[0][0]),
                float(batch_bins[0][1])
            ),
            {'id': -1.0},
            {'source': -1.0}
        ]]

        selected_indices, _, rotation_decisions, _ = model.predict(batch_inputs)
        reordered_indices = tf.gather(batch_indices[0], selected_indices[0])

        batch_size = tf.shape(batch_inputs)[0]
        batch_idx = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1),
                            [1, tf.shape(selected_indices)[1]])
        selected_indices = tf.cast(selected_indices, tf.int32)
        gather_indices = tf.stack([batch_idx, selected_indices], axis=2)
        arranged_inputs = tf.gather_nd(batch_inputs, gather_indices)

        area = compute_packing_area(arranged_inputs, reordered_indices, rotation_decisions)

        results.append({
            'file_name': batch_file_names[0].numpy().decode(),
            'packing_area': float(-area),
            'arrangement': state.S_opt.copy()
        })

        logger.info(f"Processed batch {batch_num + 1}/{len(dataset)}: "
                    f"File={results[-1]['file_name']}, Area={results[-1]['packing_area']}")

    return results


def start_prediction():
    """启动预测过程"""
    try:
        init_processing()
        results = predict_model()

        output_file = 'output/prediction_results.json'
        os.makedirs('output', exist_ok=True)  # 确保输出目录存在

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Prediction results saved to {output_file}")

        return {'status': 'success', 'message': f'Predictions completed and saved to {output_file}'}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {'status': 'error', 'message': str(e)}


def start_training():
    """启动训练过程"""
    try:
        init_processing()
        checkpoint_dir = '../training_checkpoints'
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        if latest_checkpoint:
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            train_model(resume_from=latest_checkpoint)
        else:
            logger.info("No checkpoint found. Starting training from scratch.")
            os.makedirs("../training_checkpoints", exist_ok=True)
            train_model()
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        state.is_training_started = False


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    import uvicorn

    logger.info("Starting the FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
