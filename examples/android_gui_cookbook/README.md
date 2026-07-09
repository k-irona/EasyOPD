# Android GUI 数字游戏强化学习教程

本教程涵盖：**云端环境部署** → **模型训练** → **模型测试** 三个完整流程。

---

## 1. 云端 Android 部署和游戏部署

### 1.1 游戏部署

#### Docker 部署

```bash
# 拉取并运行游戏容器
docker run -d \
  --name number-game \
  -p 8000:8000 \
  ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl:v1.4

# 访问游戏
# http://localhost:8000/number_game.html
```

#### Kubernetes 部署

```bash
# 使用提供的配置文件
kubectl apply -f examples/android_gui_cookbook/game_docker/game.yaml

# 获取外部访问地址
kubectl get svc number-game -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# 访问: http://<EXTERNAL-IP>:8000/number_game.html
```

### 1.2 Android 设备连接

#### 创建云端 Android 设备
参考文档：https://github.com/tkestack/tke-ai-playbook/pull/20

#### 连接设备并打开游戏

```bash
# 连接设备
adb connect <android_ip>:5555

# 在设备浏览器打开游戏
adb -s <android_ip>:5555 shell am start -a android.intent.action.VIEW \
  -d "http://<game_ip>:8000/number_game.html"

# 验证连接
adb devices
```

---

## 2. 模型训练

### 2.1 训练脚本说明

**核心文件**：
- `examples/qwen2_5_vl_3b_android_gui_grpo.sh` - 训练启动脚本
- `examples/format_prompt/android_gui.jinja` - 提示词模板
- `examples/reward_function/android_gui.py` - 奖励函数

**游戏规则**（由 `android_gui.jinja` 定义）：
- 🟢 绿灯：选择**最大**数字 → 位置索引 (0/1/2)
- 🔴 红灯：选择**最小**数字 → 位置索引 (0/1/2)
- 🟡 黄灯：选择**中间**数字 → 位置索引 (0/1/2)

**评分规则**（由 `android_gui.py` 实现）：
- 正确选择：`+1.0`
- 错误选择：`0.0`

### 2.2 启动训练

```bash
# 切换到 EasyOPD 根目录
cd /path/to/EasyOPD

# 运行训练脚本
bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

### 2.3 关键训练参数

脚本使用以下配置（基于 `config.yaml`，通过命令行覆盖）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `data.train_files` | `yuehua-s/numbergame@train` | 训练数据集 |
| `data.val_files` | `yuehua-s/numbergame@test` | 验证数据集 |
| `data.rollout_batch_size` | `32` | Rollout 批次大小 |
| `algorithm.kl_coef` | `0.04` | KL 散度系数 |
| `worker.actor.optim.lr` | `1e-5` | 学习率 |
| `worker.rollout.n` | `8` | 每步生成响应数 |
| `trainer.total_epochs` | `3` | 训练轮数 |
| `trainer.n_gpus_per_node` | `2` | 每节点 GPU 数 |

### 2.4 导出模型

训练完成后，检查点保存在 `checkpoints/<experiment_name>/global_step_<N>/actor`。

```bash
# 合并模型（转换为 HuggingFace 格式）
python3 scripts/model_merger.py \
  --local_dir /path/to/EasyOPD/checkpoints/<experiment_name>/global_step_35/actor

# 导出目录：checkpoints/<experiment_name>/global_step_35/actor/huggingface/
```

---

## 3. 使用 Agent 玩游戏测试模型效果

### 3.1 启动推理服务

使用 vLLM 部署训练好的模型：

```bash
vllm serve /path/to/checkpoints/<experiment_name>/global_step_35/actor/huggingface/ \
  --host 0.0.0.0 \
  --port 8000
```

### 3.2 运行 Agent 测试

**核心文件**：
- `examples/android_gui_cookbook/play_agent.py` - Agent 主程序
- `examples/android_gui_cookbook/adb_controller.py` - ADB 控制
- `examples/android_gui_cookbook/vlm_client.py` - VLM 推理客户端

#### 使用 vLLM 模型

```bash
python examples/android_gui_cookbook/play_agent.py \
  --model-type vllm \
  --api-url http://<vllm_server_ip>:8000 \
  --model-name /path/to/checkpoints/xxx/global_step_35/actor/huggingface/ \
  --devices <android_ip>:5555 \
  --episodes 5 \
  --debug
```

#### 使用 Ollama 模型

```bash
python examples/android_gui_cookbook/play_agent.py \
  --model-type ollama \
  --api-url http://localhost:11434 \
  --model-name qwen2.5vl:3b \
  --devices <android_ip1>:5555 <android_ip2>:5555 \
  --episodes 3 \
  --debug
```

### 3.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-type` | `ollama` | 模型服务类型（`ollama` 或 `vllm`） |
| `--api-url` | `http://localhost:11434` | 模型 API 地址 |
| `--model-name` | `qwen2.5vl:3b` | 模型名称或路径 |
| `--devices` | `101.43.137.83:5555` | Android 设备列表（空格分隔） |
| `--episodes` | `1` | 每个设备运行局数 |
| `--debug` | `False` | 开启调试模式（显示 VLM 输出） |
| `--screenshot-dir` | `game_screenshots` | 截图保存目录 |

### 3.4 测试流程

Agent 自动执行以下操作（每局 10 轮）：

1. **截图** - 捕获当前游戏画面
2. **VLM 推理** - 识别指示灯颜色和数字，做出决策
3. **点击卡片** - 点击选择的数字（位置 0/1/2）
4. **验证点击** - 检查卡片颜色是否改变
5. **点击下一轮** - 进入下一轮游戏

### 3.5 查看结果

测试完成后，结果保存在 `game_screenshots/<device_id>/`：

```
game_screenshots/
└── <android_ip>_5555/
    ├── round_01_<timestamp>.png           # 每轮决策前截图
    ├── round_01_after_click_<timestamp>.png  # 点击后截图
    ├── final_score_<timestamp>.png        # 最终得分截图
    └── result_<timestamp>.json            # 游戏结果（JSON）
```

**结果文件示例**：
```json
{
  "device_id": "101.43.137.83:5555",
  "timestamp": "20251123_143025",
  "total_rounds": 10,
  "final_score": 80,
  "model_type": "vllm",
  "model_name": "/path/to/model"
}
```

---

## 附录：文件结构

```
examples/
├── qwen2_5_vl_3b_android_gui_grpo.sh    # 训练脚本
├── config.yaml                           # 基础配置
├── format_prompt/
│   └── android_gui.jinja                 # 提示词模板
├── reward_function/
│   └── android_gui.py                    # 奖励函数
└── android_gui_cookbook/
    ├── README.md                         # 本文档
    ├── play_agent.py                     # Agent 主程序
    ├── adb_controller.py                 # ADB 控制器
    ├── vlm_client.py                     # VLM 客户端
    └── game_docker/
        ├── game.yaml                     # K8s 部署配置
        └── DOCKER_README.md              # Docker 详细说明
```
