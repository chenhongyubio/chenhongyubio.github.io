---
layout: post # 使用的布局（不需要改）
title: Cell Blast # 标题
subtitle: Cell blast比对注释分析流程 #副标题
date: 2020-08-26 # 时间
author: CHY # 作者
header-img: img/wallhaven-SC.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 单细胞
---

#### Cell blast 比对注释

```
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import Cell_BLAST as cb

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=200)
pd.set_option("max_rows", 6)
tf.logging.set_verbosity(0)
cb.config.N_JOBS = 4
cb.config.RANDOM_SEED = 0

## 数据库准备Preparing database
baron_human = cb.data.ExprDataSet.read_dataset("../../Datasets/data/Baron_human/data.h5")

%%capture
start_time=time.time()
models = []
for i in range(4):
    models.append(cb.directi.fit_DIRECTi(
        baron_human, genes=baron_human.uns["seurat_genes"],
        latent_dim=10, cat_dim=20, random_seed=i
    ))
print("Time elapsed: %.1fs" % (time.time() - start_time))

# 构建数据库
blast = cb.blast.BLAST(models, baron_human)
blast.save("./baron_human_blast")
del blast
blast = cb.blast.BLAST.load("./baron_human_blast")

# 数据提交
# 目前不需要对数据进行预处理
lawlor = cb.data.ExprDataSet.read_dataset("../../Datasets/data/Lawlor/data.h5")
start_time = time.time()
lawlor_hits = blast.query(lawlor)
print("Time per query: %.1fms" % (
    (time.time() - start_time) * 1000 / lawlor.shape[0]
))

lawlor_hits = lawlor_hits.reconcile_models().filter(by="pval", cutoff=0.05)
hits_dict = lawlor_hits[0:5].to_data_frames()
hits_dict.keys()
hits_dict["1st-61_S27"]
lawlor_predictions = lawlor_hits.annotate("cell_ontology_class") # 获取细胞类型预测

# 比较预测的细胞类型与ground truth结果
fig = cb.blast.sankey(
    lawlor.obs["cell_ontology_class"].values,
    lawlor_predictions.values.ravel(),
    title="Lawlor to Baron_human", tint_cutoff=2
)
```

#### DIRECTi 训练模型

```
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import Cell_BLAST as cb

np.set_printoptions(threshold=200)
pd.set_option("max_rows", 20)
tf.logging.set_verbosity(0)
cb.config.N_JOBS = 4
cb.config.RANDOM_SEED = 0

#### 数据加载
baron_human = cb.data.ExprDataSet.read_dataset("../../Datasets/data/Baron_human/data.h5") # 类似于annData，表达数据存储在exprs slot
baron_human.exprs[0:10, 0:10].toarray()

# cell meta data in obs slot
baron_human.obs.head()

# gene meta data in var slot
baron_human.var.head()

# Other unstructured data  in the uns slot
baron_human.uns.keys()
baron_human.uns["seurat_genes"]

#### 基因选择
%%capture
selected_genes, axes = baron_human.find_variable_genes(grouping="donor")
selected_genes
np.setdiff1d(selected_genes, baron_human.uns["seurat_genes"]).size, \
np.setdiff1d(baron_human.uns["seurat_genes"], selected_genes).size

#### 无监督降维
%%capture
model = cb.directi.fit_DIRECTi(
    baron_human, genes=selected_genes,
    latent_dim=10, cat_dim=20
)
baron_human.latent = model.inference(baron_human)
ax = baron_human.visualize_latent("cell_ontology_class")

#### 模型保存
model.save("./baron_human_model")
model.close()
del model
model = cb.directi.DIRECTi.load("./baron_human_model")
```
