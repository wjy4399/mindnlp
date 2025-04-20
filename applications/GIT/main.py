import json
import sys

captions = [{"file_name": "ronaldo.jpeg", "text": "Ronaldo with Portugal at the 2018 World Cup"},
{"file_name": "messi.jpeg", "text": "Messi with Argentina at the 2022 FIFA World Cup"},
{"file_name": "zidane.jpeg", "text": "Zinédine Zidane pendant la finale de la Coupe du monde 2006."},
{"file_name": "maradona.jpeg", "text": "Maradona after winning the 1986 FIFA World Cup with Argentina"},
{"file_name": "ronaldo_.jpeg", "text": "Ronaldo won La Liga in his first season and received the Pichichi Trophy in his second."},
{"file_name": "pirlo.jpeg", "text": "Pirlo with Juventus in 2014"},]

# path to the folder containing the images
root = "Toy_dataset/"

# add metadata.jsonl file to this folder
with open(root + "metadata.jsonl", 'w') as f:
    for item in captions:
        f.write(json.dumps(item) + "\n")
from datasets import load_dataset 

dataset = load_dataset("imagefolder", data_dir=root, split="train")
example = dataset[0]
image = example["image"]
width, height = image.size

# MindSpore实现图像描述数据集
class ImageCaptioningDataset:
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.encoded_inputs_list = []
        self.__get_encoded_inputs_list__()
    
    def __get_encoded_inputs_list__(self):
        for idx in range(self.__len__()):
            item = self.dataset[idx]
            # GitProcessor可以同时处理图像和文本
            try:
                encoding = self.processor(
                    images=item["image"], 
                    text=item["text"], 
                    padding="max_length", 
                    max_length=512,  # 设置合适的长度，小于config中的max_position_embeddings
                    return_tensors="ms"  # 使用"ms"表示MindSpore tensors
                )
                
                # remove batch dimension
                for k, v in encoding.items():
                    encoding[k] = v.squeeze()
                
                # 直接返回一个tuple，而不是dict
                self.encoded_inputs_list.append(tuple(encoding.values()))
            except Exception as e:
                print(f"处理样本 {idx} 时出错: {e}")
                # 尝试使用更兼容的方式处理
                try:
                    # 分别处理图像和文本
                    pixel_values = self.processor.image_processor(item["image"], return_tensors="ms").pixel_values.squeeze()
                    text_inputs = self.processor.tokenizer(
                        item["text"], 
                        padding="max_length", 
                        max_length=512, 
                        return_tensors="ms"
                    )
                    
                    # 移除批次维度
                    for k, v in text_inputs.items():
                        text_inputs[k] = v.squeeze()
                    
                    # 合并处理结果
                    inputs = {}
                    inputs.update(text_inputs)
                    inputs["pixel_values"] = pixel_values
                    
                    self.encoded_inputs_list.append(tuple(inputs.values()))
                except Exception as inner_e:
                    print(f"备用处理方法也失败: {inner_e}")
                continue

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.encoded_inputs_list[idx]

from transformers import AutoProcessor
from mindnlp.transformers import GitConfig, GitForCausalLM, GitProcessor
from mindnlp.engine import TrainingArguments, Trainer
import numpy as np
from sklearn.model_selection import train_test_split

# 使用MindNLP的GitProcessor
try:
    processor = GitProcessor.from_pretrained("microsoft/git-base")
except Exception as e:
    print(f"尝试使用MindNLP的GitProcessor失败: {e}")
    # 回退到transformers的处理器
    processor = AutoProcessor.from_pretrained("microsoft/git-base")

# 创建数据集实例
full_dataset = ImageCaptioningDataset(dataset, processor)

# 分割数据集为训练集和评估集
train_indices, eval_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)

class SubsetDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        # 添加列名属性，用于与mindnlp的trainer兼容
        self.column_names = ["input_ids", "attention_mask", "pixel_values"]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    # 添加get_col_names方法，以兼容mindnlp的trainer
    def get_col_names(self):
        return self.column_names
    
    # 添加map方法，兼容mindnlp的数据处理流程
    def map(self, fn, input_columns=None, output_columns=None):
        # 这里简单返回自身，因为我们的数据已经预处理好了
        return self
    
    # 添加batch方法，兼容mindnlp的数据处理流程
    def batch(self, batch_size, drop_last=False, num_workers=None):
        # 由于我们的数据已经以元组形式存储，这里简单返回自身
        return self

train_dataset = SubsetDataset(full_dataset, train_indices)
eval_dataset = SubsetDataset(full_dataset, eval_indices)

# 使用MindNLP自己的GIT模型实现
from mindnlp.transformers import GitConfig, GitForCausalLM

# 创建GIT模型配置
config = GitConfig.from_pretrained("microsoft/git-base")
model = GitForCausalLM(config)

# 如果需要初始化部分权重，可以尝试以下代码（可选）
# from mindnlp.transformers.utils import convert_pt_to_ms
# pytorch_model_path = processor.download_model("microsoft/git-base", "pytorch_model.bin")
# ms_weights = convert_pt_to_ms(pytorch_model_path)
# load_param_into_net(model, ms_weights)

# 定义评估指标计算函数
def compute_metrics(eval_pred):
    try:
        logits, labels = eval_pred
        # 由于是生成任务，我们简化评估逻辑，仅返回损失值
        # 实际应用中可以添加BLEU、ROUGE等文本生成评估指标
        loss_value = float(np.mean([np.mean((l1 - l2) ** 2) for l1, l2 in zip(logits, labels) if len(l1) > 0 and len(l2) > 0]))
        return {"loss": loss_value}
    except Exception as e:
        print(f"计算评估指标时出错: {e}")
        return {"loss": 999.0}  # 返回一个大的损失值，表示评估失败

# 在训练前检查数据集格式
print(f"训练集大小: {len(train_dataset)}")
print(f"评估集大小: {len(eval_dataset)}")

# 检查第一个样本的格式
if len(train_dataset) > 0:
    try:
        first_sample = train_dataset[0]
        print(f"第一个样本类型: {type(first_sample)}")
        if isinstance(first_sample, tuple):
            print(f"元组长度: {len(first_sample)}")
            for i, item in enumerate(first_sample):
                print(f"第 {i} 项: 类型={type(item)}, 形状={item.shape if hasattr(item, 'shape') else '未知'}")
    except Exception as e:
        print(f"检查样本时出错: {e}")

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./git_image_captioning_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    push_to_hub=False,
)

# 由于Trainer与我们的自定义数据集不兼容，我们直接实现训练循环
import mindspore as ms
from mindspore import nn
from mindspore.train.callback import LossMonitor
from tqdm import tqdm
import torch

# 检查是否支持PyTorch
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 将模型转换为PyTorch模型
    # 注意：这里假设model是一个兼容PyTorch的模型或可以转换为PyTorch模型
    if hasattr(model, "to"):
        model.to(device)
    
    # 创建PyTorch优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_args.learning_rate))
    
    # 定义训练步骤
    def train_step(data_batch):
        optimizer.zero_grad()
        
        input_ids, attention_mask, pixel_values = data_batch
        
        # 转换为PyTorch张量
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids.asnumpy() if hasattr(input_ids, 'asnumpy') else input_ids)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask.asnumpy() if hasattr(attention_mask, 'asnumpy') else attention_mask)
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values.asnumpy() if hasattr(pixel_values, 'asnumpy') else pixel_values)
        
        # 移动到正确的设备
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values = pixel_values.to(device)
        
        # 构建输入字典
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": input_ids  # 使用input_ids作为标签
        }
        
        # 前向传播
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 计算梯度并更新
        loss.backward()
        optimizer.step()
        
        return loss.item()
except ImportError:
    print("PyTorch不可用，尝试使用MindSpore原生接口")
    # 如果PyTorch不可用，尝试使用MindSpore原生接口
    # 创建MindSpore优化器
    try:
        # 将模型参数转换为Parameter列表
        from mindspore.common.parameter import Parameter
        params = []
        for name, param in model.parameters_and_names():
            if param.requires_grad:
                params.append(Parameter(param.data, name=name))
        
        optimizer = nn.Adam(params, learning_rate=float(training_args.learning_rate))
        
        # 定义训练步骤
        def train_step(data_batch):
            # MindSpore实现...
            pass
    except Exception as e:
        print(f"创建优化器失败: {e}")
        raise

# 创建训练循环
print("开始训练...")
for epoch in range(min(int(training_args.num_train_epochs), 5)):  # 减少训练轮数以便快速测试
    print(f"Epoch {epoch+1}/{int(training_args.num_train_epochs)}")
    
    # 训练模式
    if hasattr(model, "train"):
        model.train()
    else:
        model.set_train(True)
    
    epoch_loss = 0.0
    batch_count = 0
    
    # 训练步骤
    for i in tqdm(range(len(train_dataset))):
        data = train_dataset[i]
        # 假设data是一个包含input_ids, attention_mask, pixel_values的元组
        
        try:
            # 使用train_step函数进行训练
            loss_value = train_step(data)
            
            epoch_loss += loss_value
            batch_count += 1
            
            if i % 1 == 0:  # 每个样本打印一次损失
                print(f"Batch {i}, Loss: {loss_value:.4f}")
        except Exception as e:
            print(f"训练样本 {i} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算平均损失
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    # 评估步骤
    if epoch % 1 == 0:  # 每个epoch评估一次
        print("进行评估...")
        # 设置为评估模式
        if hasattr(model, "eval"):
            model.eval()
        else:
            model.set_train(False)
        
        eval_loss = 0.0
        eval_batch_count = 0
        
        for i in range(len(eval_dataset)):
            data = eval_dataset[i]
            
            try:
                # 转换为PyTorch张量
                input_ids, attention_mask, pixel_values = data
                
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids.asnumpy() if hasattr(input_ids, 'asnumpy') else input_ids)
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask.asnumpy() if hasattr(attention_mask, 'asnumpy') else attention_mask)
                if not isinstance(pixel_values, torch.Tensor):
                    pixel_values = torch.tensor(pixel_values.asnumpy() if hasattr(pixel_values, 'asnumpy') else pixel_values)
                
                # 移动到正确的设备
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                pixel_values = pixel_values.to(device)
                
                # 使用PyTorch的no_grad来评估
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=input_ids
                    )
                
                loss_value = outputs.loss.item()
                eval_loss += loss_value
                eval_batch_count += 1
            except Exception as e:
                print(f"评估样本 {i} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if eval_batch_count > 0:
            avg_eval_loss = eval_loss / eval_batch_count
            print(f"评估损失: {avg_eval_loss:.4f}")

# 保存模型
save_path = "./git_image_captioning_final"
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 根据不同的框架保存模型
try:
    if 'torch' in sys.modules:
        # 使用PyTorch保存
        torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    else:
        # 使用MindSpore保存
        ms.save_checkpoint(model, os.path.join(save_path, "model.ckpt"))
    print(f"模型已保存到 {save_path}")
except Exception as e:
    print(f"保存模型时出错: {e}")
    import traceback
    traceback.print_exc()

# 用训练好的模型生成图像描述
def generate_caption(image_path):
    from PIL import Image
    import mindspore as ms
    
    # 加载图像
    image = Image.open(image_path)
    
    try:
        # 使用处理器处理图像
        inputs = processor(images=image, return_tensors="ms")
        
        # 设置为评估模式
        if hasattr(model, "eval"):
            model.eval()
        else:
            model.set_train(False)
        
        # 如果是PyTorch模型，需要转换张量
        if 'torch' in sys.modules:
            # 转换为PyTorch张量
            pixel_values = torch.tensor(inputs.pixel_values.asnumpy() if hasattr(inputs.pixel_values, 'asnumpy') else inputs.pixel_values)
            
            # 移动到正确的设备
            pixel_values = pixel_values.to(device)
            
            # 创建输入字典
            pytorch_inputs = {"pixel_values": pixel_values}
            
            try:
                # 使用PyTorch的no_grad来生成
                with torch.no_grad():
                    # 尝试使用标准生成接口
                    generated_ids = model.generate(
                        **pytorch_inputs,
                        max_length=50,
                        num_beams=4,
                        early_stopping=True
                    )
                
                # 解码生成的文本
                generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"PyTorch生成接口失败: {e}，尝试备用方法")
                
                # 备用方法：直接前向传播并贪婪解码
                with torch.no_grad():
                    outputs = model(**pytorch_inputs)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                # 简单的贪婪解码
                generated_ids = torch.argmax(logits, dim=-1)
                
                # 解码生成的文本
                generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        else:
            # 使用MindSpore接口
            try:
                # 尝试使用标准生成接口
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True
                )
                
                # 解码生成的文本
                generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"标准生成接口失败: {e}，尝试备用方法")
                
                # 备用方法：直接前向传播并贪婪解码
                outputs = model(pixel_values=inputs.pixel_values)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                # 简单的贪婪解码
                if hasattr(ms.ops, 'argmax'):
                    generated_ids = ms.ops.argmax(logits, axis=-1)
                else:
                    import numpy as np
                    logits_np = logits.asnumpy() if hasattr(logits, 'asnumpy') else logits.numpy()
                    generated_ids = np.argmax(logits_np, axis=-1)
                
                # 解码生成的文本
                generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        if not generated_caption:
            generated_caption = "无法生成描述。模型训练可能不足。"
    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        generated_caption = "处理图像时出错"
    
    return generated_caption

# 测试模型
try:
    test_image_path = root + "ronaldo.jpeg"  # 示例测试图像
    print(f"测试图像路径: {test_image_path}")
    generated_caption = generate_caption(test_image_path)
    print(f"生成的图像描述: {generated_caption}")
except Exception as e:
    print(f"测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()