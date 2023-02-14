# 1. 数据预处理

## 1.1 数据概况

- 标签平衡性
- 文本长度
- 是否存在繁体、英文、表情、unicode编码、**换行符**
- unicode编码难题：\xa0、\u2002、\u2003


可视化展现

## 1.2 文本向量化

- 将评论、评价维度转为向量——word2vec

- 构造评价维度标签集y

- LSAN模型所需输入：评论文本索引、标签one-hot、评论文本预训练词向量、评价维度预训练词向量

  **输入X——(batch_size, seq_len)**

  <img src="F:\研究生\论文\LSAN输入X.jpg" alt="LSAN输入X" style="zoom:50%;" />

  **输入Y——(batch_size, num_labels)**

  <img src="F:\研究生\论文\LSAN输入Y.jpg" alt="LSAN输入Y" style="zoom:50%;" />

  **标签embedding——(num_labels, embedding_size)**

  <img src="F:\研究生\论文\LSAN标签embedding.jpg" alt="LSAN标签embedding" style="zoom:50%;" />

  **评论文本词向量——(vocab_size, embedding_size)**

*Note: For LSAN model, embedding_size=300 in default*


# 2. 评价维度识别——多标签文本分类

## 2.1 尝试：

 - 在768特征维度上进行融合，尝试拼接、相加、相乘，比较效果
 - 在num_classes维度上进行融合，同样尝试拼接、相加、相乘，比较效果


LSAN实验结果
v1: epoch_1准确率较高(80%左右)，但后续效果越来越差，初步判断可能是学习率过大的问题，考虑增加学习率衰减策略
v2: 增加学习率衰减，每一个epoch后学习率乘以0.7，10个epoch的效果相对好
后期可尝试改变max_seq_len
