#!/usr/bin/env python3
"""
真正的多GPU多进程批量采样脚本
每个GPU一个独立进程，避免GPU内存混乱
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import multiprocessing as mp
from queue import Queue
import traceback
import torch
from diffusers import CogVideoXPipeline,CogVideoXDPMScheduler,CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

# 简化的配置类
class SimpleConfig:
    def __init__(self, config_file: str):
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 必需配置
        self.experiment_name = data.get("experiment_name", "video_generation")
        self.gpu_ids = data.get("gpu_ids", [0])
        self.naming_prompt_file = data["naming_prompt_file"]
        self.sampling_prompt_file = data["sampling_prompt_file"]
        self.output_dir = data.get("output_dir", "./output")
        self.cache_dir = data.get("cache_dir", "/workspace/.cache/huggingface/")
        
        # 可选配置
        self.lora_path = data.get("lora_path", None)
        self.use_sparse_attention = data.get("use_sparse_attention", False)
        self.num_inference_steps = data.get("num_inference_steps", 8)
        self.guidance_scale = data.get("guidance_scale", 1.0)
        self.batch_size = data.get("batch_size", 4)
        self.max_prompts = data.get("max_prompts", None)
        self.start_index = data.get("start_index", 0)
        self.videos_per_prompt = data.get("videos_per_prompt", 5)
        self.base_seed = data.get("base_seed", 42)
        self.transformer_weight = data.get("transformer_weight",None)
        # 固定配置
        self.model_name = "THUDM/CogVideoX-5b"
        self.num_frames = 49
        self.fps = 8
    
    def to_dict(self):
        return self.__dict__

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_prompts(config: SimpleConfig) -> Tuple[List[str], List[str]]:
    """加载提示词 - 返回 (命名用提示词, 采样用提示词)"""
    # 加载命名用的简短提示词
    with open(config.naming_prompt_file, 'r', encoding='utf-8') as f:
        naming_prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # 加载采样用的增强提示词
    with open(config.sampling_prompt_file, 'r', encoding='utf-8') as f:
        sampling_prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # 确保两个文件长度一致
    if len(naming_prompts) != len(sampling_prompts):
        min_len = min(len(naming_prompts), len(sampling_prompts))
        naming_prompts = naming_prompts[:min_len]
        sampling_prompts = sampling_prompts[:min_len]
    
    # 应用起始索引和最大数量限制
    end_index = len(naming_prompts)
    if config.max_prompts:
        end_index = min(config.start_index + config.max_prompts, len(naming_prompts))
    
    naming_prompts = naming_prompts[config.start_index:end_index]
    sampling_prompts = sampling_prompts[config.start_index:end_index]
    
    return naming_prompts, sampling_prompts

def gpu_worker(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, config_dict: dict):
    """GPU工作进程"""
    logger = setup_logging()
    logger.info(f"GPU {gpu_id} 工作进程启动 (PID: {os.getpid()})")
    
    try:
        # 重构config对象 
        config = SimpleConfig.__new__(SimpleConfig)
        config.__dict__.update(config_dict)
        
        # 设置GPU设备
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        logger.info(f"GPU {gpu_id}: 初始化Pipeline")
        
        # 初始化pipeline
        pipe = CogVideoXPipeline.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
            vision="main"
        )
                
        # 加载LoRA
        if config.lora_path:
            logger.info(f"GPU {gpu_id}: 加载LoRA {config.lora_path}")
            pipe.load_lora_weights(config.lora_path)

        if config.transformer_weight:
            logger.info(f"GPU {gpu_id}: 加载transformer weight: {config.transformer_weight}")
            pipe.transformer = CogVideoXTransformer3DModel.from_pretrained(config.transformer_weight,torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)
        pipe.vae.enable_tiling()
        # 应用稀疏注意力
        if config.use_sparse_attention:
            try:
                from modify_cogvideo import set_block_sparse_attn_cogvideox
                set_block_sparse_attn_cogvideox(pipe.transformer)
                logger.info(f"GPU {gpu_id}: 应用稀疏注意力")
            except ImportError as e:
                logger.error(f"GPU {gpu_id}: 找不到sparse attention模块 - {e}")
                logger.error(f"GPU {gpu_id}: 请检查 modify_cogvideo.py 是否存在")
                # 不要中断，继续运行但不使用稀疏注意力
            except Exception as e:
                logger.error(f"GPU {gpu_id}: 稀疏注意力初始化失败 - {e}")
                exit(-2)
                # 不要中断，继续运行但不使用稀疏注意力
        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        # 检查内存
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        logger.info(f"GPU {gpu_id}: 初始化完成，内存使用 {memory_allocated:.2f}GB")
        
        # 开始处理任务
        while True:
            try:
                batch_data = task_queue.get(timeout=1)
                if batch_data is None:  # 结束信号
                    logger.info(f"GPU {gpu_id}: 收到结束信号")
                    break
                
                logger.info(f"GPU {gpu_id}: 开始处理 {len(batch_data)} 个视频")
                
                # 处理批次
                batch_results = process_batch(pipe, batch_data, config, gpu_id, device)
                
                # 发送结果
                result_queue.put((gpu_id, batch_results))
                
            except mp.queue.Empty:
                continue
            except Exception as e:
                logger.error(f"GPU {gpu_id}: 处理出错: {e}")
                result_queue.put((gpu_id, [{'success': False, 'error': str(e)}]))
        
        logger.info(f"GPU {gpu_id}: 工作进程结束")
        
    except Exception as e:
        logger.error(f"GPU {gpu_id}: 初始化失败: {e}")
        result_queue.put((gpu_id, [{'success': False, 'error': str(e)}]))

def process_batch(pipe, batch_data: List[Tuple], config: SimpleConfig, gpu_id: int, device: str) -> List[Dict]:
    """处理一个批次的视频生成"""
    logger = setup_logging()
    
    batch_prompts = []
    batch_generators = []
    batch_info = []
    
    for sampling_prompt, output_path, prompt_index, video_index in batch_data:
        batch_prompts.append(sampling_prompt)
        
        seed = config.base_seed + prompt_index * 1000 + video_index
        generator = torch.Generator(device=device).manual_seed(seed)
        batch_generators.append(generator)
        
        batch_info.append({
            'output_path': output_path,
            'prompt_index': prompt_index,
            'video_index': video_index,
            'seed': seed
        })
    
    try:
        start_time = time.time()
        
        # 批量生成
        batch_results = pipe(
            prompt=batch_prompts,
            num_videos_per_prompt=1,
            num_inference_steps=config.num_inference_steps,
            num_frames=config.num_frames,
            guidance_scale=config.guidance_scale,
            generator=batch_generators,
        )
        
        generation_time = time.time() - start_time
        avg_time = generation_time / len(batch_data)
        
        results = []
        for i, info in enumerate(batch_info):
            try:
                # 保存视频
                video_frames = batch_results.frames[i]
                os.makedirs(os.path.dirname(info['output_path']), exist_ok=True)
                # print("导出视频中...")
                export_to_video(video_frames, info['output_path'], fps=config.fps)
                # print("视频导出成功！")
                results.append({
                    'success': True,
                    'output_path': info['output_path'],
                    'generation_time': avg_time,
                    'prompt_index': info['prompt_index'],
                    'video_index': info['video_index'],
                    'seed': info['seed']
                })
                
            except Exception as e:
                results.append({
                    'success': False,
                    'error': f"保存失败: {str(e)}",
                    'prompt_index': info['prompt_index'],
                    'video_index': info['video_index'],
                    'output_path': info['output_path']
                })
                traceback.print_exc()
        
        return results
        
    except Exception as e:
        logger.error(f"GPU {gpu_id}: 批量生成失败: {e}")
        return [{'success': False, 'error': str(e)} for _ in batch_info]

def main():
    parser = argparse.ArgumentParser(description="多进程多GPU CogVideoX批量采样")
    parser.add_argument("--config", default="simple_config.json", help="配置文件路径")
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # 加载配置
    config = SimpleConfig(args.config)
    logger.info(f"实验名称: {config.experiment_name}")
    
    # 创建输出目录
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载提示词
    naming_prompts, sampling_prompts = load_prompts(config)
    logger.info(f"加载了 {len(naming_prompts)} 个提示词")
    
    # 准备任务
    tasks = []
    for i, (naming_prompt, sampling_prompt) in enumerate(zip(naming_prompts, sampling_prompts)):
        prompt_idx = config.start_index + i
        
        for video_idx in range(config.videos_per_prompt):
            clean_prompt = naming_prompt.replace('/', '_').replace('\\', '_').replace('|', '_')
            filename = f"{clean_prompt}-{video_idx}.mp4"
            output_path = str(output_dir / filename)
            
            if os.path.exists(output_path):
                logger.info(f"跳过已存在文件: {filename}")
                continue
                
            tasks.append((sampling_prompt, output_path, prompt_idx, video_idx))
    
    if not tasks:
        logger.info("没有需要处理的任务")
        return
    
    # 创建批次
    def create_batches(tasks, batch_size):
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    
    batches = list(create_batches(tasks, config.batch_size))
    
    logger.info(f"开始多进程采样:")
    logger.info(f"  任务数: {len(tasks)} 个视频")
    logger.info(f"  批次数: {len(batches)} 个批次")
    logger.info(f"  GPU数: {len(config.gpu_ids)}")
    logger.info(f"  批量大小: {config.batch_size}")
    
    # 创建进程间通信队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 将所有批次放入任务队列
    for batch in batches:
        task_queue.put(batch)
    
    # 启动GPU工作进程
    processes = []
    for gpu_id in config.gpu_ids:
        p = mp.Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue, config.__dict__))
        p.start()
        processes.append(p)
        logger.info(f"启动GPU {gpu_id} 工作进程 (PID: {p.pid})")
    
    # 收集结果
    completed = 0
    failed = 0
    start_time = time.time()
    
    for _ in range(len(batches)):
        gpu_id, batch_results = result_queue.get()
        
        for result in batch_results:
            if result.get('success'):
                completed += 1
                logger.info(f"✓ 完成 {completed}/{len(tasks)}: {os.path.basename(result['output_path'])} "
                           f"(GPU {gpu_id}, {result['generation_time']:.1f}s)")
            else:
                failed += 1
                error_msg = result.get('error', '未知错误')
                logger.error(f"✗ 失败 {failed}: GPU {gpu_id} - {error_msg}")
                if 'prompt_index' in result:
                    logger.error(f"  失败任务: prompt_index={result['prompt_index']}, video_index={result.get('video_index', 'N/A')}")
                if 'output_path' in result:
                    logger.error(f"  输出路径: {result['output_path']}")
    
    # 发送结束信号
    for _ in config.gpu_ids:
        task_queue.put(None)
    
    # 等待所有进程结束
    for p in processes:
        p.join()
    
    # 统计结果
    total_time = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"多进程采样完成!")
    logger.info(f"总时间: {total_time:.1f} 秒")
    logger.info(f"成功: {completed} 个")
    logger.info(f"失败: {failed} 个")
    if completed > 0:
        logger.info(f"平均每个视频: {total_time/completed:.1f} 秒")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()