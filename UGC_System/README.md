# UGC System

当前项目已经调整为纯文件工作流，不使用 MySQL。实验状态、日志、粗图数据和测试结果都直接落盘在 `outputs/` 下。
当前结果页只读取新实验保存的结构化结果文件，不再兼容旧实验的临时补算逻辑。

## 当前项目结构

```text
UGC_System/
├─ app.py                  # Flask 入口与页面路由
├─ config.py               # 路径配置
├─ requirements.txt        # 依赖列表
├─ README.md               # 项目说明
│
├─ algorithms/             # 后端流程与服务层
│  ├─ pipeline.py          # 总流程编排
│  ├─ alpha_service.py     # alpha 计算封装
│  ├─ preprocess_service.py# 预处理封装
│  ├─ calibrate_service.py # bin width 标定
│  ├─ ugc_service.py       # 调用 UGC 主程序并保存粗化产物
│  ├─ gnn_service.py       # 前端可选模型/策略列表
│  └─ utils.py             # 通用工具函数
│
├─ scripts/                # 原始脚本保留区
│  ├─ get_alpha/
│  ├─ data_process/
│  ├─ calibrate_ugc.py
│  └─ UGC.py
│
├─ templates/              # Flask 页面模板
│  ├─ base.html
│  ├─ upload.html
│  ├─ running.html
│  ├─ result.html
│  └─ gnn_test.html
│
├─ static/
│  ├─ css/style.css        # 页面样式
│  └─ js/
│     ├─ graph_view.js     # Cytoscape 图联动逻辑
│     └─ result_chart.js   # ECharts 结果图逻辑
│
├─ data/                   # 内置数据集与预处理产物
├─ uploads/                # 用户上传数据集
├─ outputs/                # 每次实验的结果目录
├─ logs/                   # 预留日志目录
└─ models/                 # 当前未使用，仅保留空占位说明
```

## 系统页面

现在系统有 4 个主要页面：

1. 上传页
   用户上传文件或选择内置数据集，并设置策略、是否无向化、target ratio 等参数。

2. 运行页
   展示当前阶段、日志和中间结果，例如 alpha、bin width、压缩率。

3. 结果页
   展示粗化结果、真实 coarse graph、supernode 详情、原图映射联动和结果图表。

4. GNN 测试页
   在已生成的粗图基础上重新选择 GNN 模型和训练轮数，再做测试。

## 后端主流程

统一流程由 `algorithms/pipeline.py` 驱动，顺序如下：

1. 计算 alpha
2. 数据预处理
3. 标定 bin width
4. 运行 UGC
5. 在 UGC 运行时直接保存 coarse graph、mapping 与 coarse dataset
6. 写入 `outputs/<run_id>/`

## 当前支持的多标签策略

- `v1`: 第一标签策略
- `v2`: 标签集合交集策略
- `v3`: 位置对齐匹配策略
- `v4`: Jaccard 阈值策略

这些策略分别通过 `scripts/get_alpha/` 和 `scripts/data_process/` 中的原始脚本实现，再由 `algorithms/` 中的 service 层统一封装。

## 输出目录说明

每次实验都会在 `outputs/<run_id>/` 下生成部分或全部文件：

- `state.json`：运行状态与阶段日志
- `summary.json`：本次实验摘要
- `ugc_stdout.log`：UGC 标准输出
- `ugc_stderr.log`：UGC 错误输出
- `coarse_graph.json`：真实 coarse graph，包含默认展示边和全量 `all_edges`
- `mapping.json`：结果页使用的原图/粗图节点映射与边映射
- `metadata.json`：粗化结果元数据
- `supernode_members.json`：supernode 到原始节点映射
- `exports/pt/`：PT 格式导出
- `exports/text/`：边集/标签/特征文本导出
- `gnn_tests/*.json`：额外 GNN 测试结果

## 运行方式

建议在 `ugc_cuda` 环境中运行：

```powershell
C:\Users\86153\.conda\envs\ugc_cuda\python.exe app.py
```

如果只想走命令行流程：

```powershell
C:\Users\86153\.conda\envs\ugc_cuda\python.exe -m algorithms.pipeline --dataset facebook --strategy v4 --train-epochs 1
```

## 当前结果页依赖

结果页默认依赖以下两个核心文件：

- `coarse_graph.json`
- `mapping.json`

其中：

- `coarse_graph.json` 用于 coarse graph 绘制、supernode 详情和全量边百分位筛选
- `mapping.json` 用于原图与粗图的节点联动、边联动和详情展示
