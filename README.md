# 澳门路牌识别 · 校内人工智能考试实践

使用 PyTorch / ResNet-50 进行 89 类澳门路牌图像分类，并通过 Kaggle 提交预测结果。

代码由 AI 辅助生成；本人参与检查数据流程、发现裁剪问题、调整预处理并进行训练与提交。本归档不作为独立手写代码的证明。

## 学习经历与问题修正

在人工智能基础及 YOLO 相关知识考核中积累概念基础后，将所学用于校内路牌识别考试项目。知识考核与本项目为两项经历。

初期使用整张图片训练，训练与验证损失表现不理想。检查数据预处理流程后，调整为先裁剪路牌区域，再进行分类训练，随后识别表现改善。也经历了家中训练、学校运行时因环境不同而出现的问题，由此更加重视运行依赖、路径和推理入口。

以上为参与者对实践过程的回顾；没有保留下完整原始训练日志，不能据此补画损失曲线或声称严格的对照实验结论。

## 最终结果

**Kaggle Private Score：0.95000**。截图同时显示 Best Score 为 0.95000；未核实竞赛官方评分定义，故不将分数直接改写成“准确率 95%”。

![Kaggle 结果原截图](images/kaggle-score.png)

## 保存下来的部署流程

1. 按输入记录的路牌区域裁剪图像。
2. 保持比例填充为正方形，调整为 256 × 256，进行归一化。
3. 使用单个 ResNet-50 分类模型推理 89 类标签。
4. 导出表头为 `img,class` 的 `submission.csv`。

`classes.txt` 保存类别标签；`src/ensemble_infer.py` 虽沿用旧名称，但当前入口配置为 **single model**，不应据文件名将其描述为多模型集成。

## 运行入口

```bash
python -m pip install -r requirements_cpu.txt
python src/ensemble_infer.py --checkpoint checkpoints/final_single_resnet50.pt --input_csv /path/to/test.csv --image_dir /path/to/test --output submission.csv --debug_csv debug_predictions.csv --device cpu
```

在 Kaggle 上的原入口为 `run_kaggle.py`，它会寻找 `/kaggle/input` 中的项目、模型及测试输入，再调用推理程序。

## 文件与复现边界

- 保留原归档中的源程序与配置，包含历史实验辅助程序；并非所有配置都被最终入口使用。
- 公开源码目录不含模型权重、训练图片或测试图片。权重文件名为 `final_single_resnet50.pt`，需另行取得后放入 `checkpoints/`。
- 最终截图分数与归档内本地验证指标属于不同记录，不能混用。
- 原训练图片、完整训练日志及最初 Word 报告尚未收录。
- 当前整理进行了语法、结构与资料核对，没有重跑模型训练或端到端推理。

## 代码整理

本次改善 Kaggle 入口的局部变量命名，并将恒为 `False` 的条件表达式写明。保留接口、模型配置与预处理行为，通过 84 个 Python 文件的语法解析及整理前后语法树对照。没有重跑训练或模型推理。
