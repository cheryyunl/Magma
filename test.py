#!/usr/bin/env python3
"""
Magma Video Benchmark Testing Script - 严格按照原始prompt格式
"""

import os
import json
import re
from collections import defaultdict
import torch
from torchvision.transforms.functional import to_pil_image
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset
import pandas as pd

class VideoBenchmarkTester:
    def __init__(self, model_name="microsoft/Magma-8B"):
        self.dtype = torch.bfloat16
        
        print(f"🔄 加载模型: {model_name}")
        print(f"🖥️  可用GPU: {torch.cuda.device_count()}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=self.dtype,
            device_map="auto",  # 自动分配到多个GPU
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("✅ 模型已分布到多个GPU")
        
        self.results = []

    def load_video(self, video_path, max_frames_num=32):
        """加载视频 - 完全按照原始代码"""
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def inference_single_sample(self, example, idx):
        """推理单个样例 - 严格按照原始Magma prompt"""
        try:
            video_path = example["question_path"]
            print(f"Processing {idx} {video_path}")
            
            # 处理视频
            frames = self.load_video(video_path, 32)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            images = [to_pil_image(frame) for frame in frames]
            
            # 构造输入 - 完全按照原始代码
            contexts = "Answer the question in this video."
            
            # 原始Magma prompt格式
            convs = [
                {"role": "user", "content": ''.join(["<image>\n"]*len(images)) + contexts},
            ]
            convs = [
                {"role": "system", "content": "You are agent that can see, talk and act."},            
            ] + convs            
            
            prompt = self.processor.tokenizer.apply_chat_template(
                convs,
                tokenize=False,
                add_generation_prompt=True
            )

            # 原始代码的条件判断
            if self.model.config.mm_use_image_start_end:
                prompt = prompt.replace("<image>", "<image_start><image><image_end>")
            
            inputs = self.processor(images=images, texts=prompt, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
            inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
            
            # 将inputs移动到模型所在的设备
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)
            
            # 推理参数 - 按照原始代码
            gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0,
                "do_sample": False
            }

            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    temperature=gen_kwargs["temperature"],
                    do_sample=gen_kwargs["do_sample"],
                )
                output = output[:, inputs["input_ids"].shape[-1] :]
                response = self.processor.decode(output[0], skip_special_tokens=True).strip()

            return {
                "idx": idx,
                "video_path": video_path,
                "ground_truth": example.get("ground_truth", ""),
                "prediction": response,
                "question_text": example.get("question_text", "")
            }
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return {
                "idx": idx,
                "video_path": video_path,
                "ground_truth": example.get("ground_truth", ""),
                "prediction": "",
                "question_text": example.get("question_text", "")
            }

    def run_benchmark(self):
        """运行morse-500测试"""
        print("🔄 加载morse-500数据集")
        dataset = load_dataset('video-reasoning/morse-500')
        
        print(f"🚀 开始处理 {len(dataset['test'])} 个样本")
        
        for i, example in tqdm(enumerate(dataset['test']), desc="Processing", total=len(dataset['test'])):
            result = self.inference_single_sample(example, i)
            self.results.append(result)
        
        # 保存结果
        results_df = pd.DataFrame(self.results)
        results_df.to_csv("morse_500_magma_results.csv", index=False)
        print(f"✅ 结果已保存到 morse_500_magma_results.csv")

def main():
    # 设置多GPU环境
    print(f"🖥️  总GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    tester = VideoBenchmarkTester("microsoft/Magma-8B")
    tester.run_benchmark()

if __name__ == "__main__":
    main()