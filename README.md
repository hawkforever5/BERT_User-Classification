# 微博发声人鉴别

[**中文说明**](https://github.com/hawkforever5/BERT_User-Classification/blob/main/README.md) | [**English**](https://github.com/hawkforever5/BERT_User-Classification/blob/main/README_EN.md)

利用微博的用户数据对用户进行分类。

原数据共有24个分类：'超话粉丝大咖': 0, '公务员': 1, '大V名人': 2, '党委': 3, '国防军委': 4, '基层组织': 5, '政府': 6, '检验检测': 7, '媒体': 8, '民主党派': 9, '明星红人': 10, '企事业单位': 11, '赛事活动': 12, '社会组织': 13, '社区组织': 14, '司法机关': 15, '外国政府机构': 16, '网民': 17, '行业专家': 18, '学校': 19, '研究机构': 20, '演艺娱乐明星': 21, '政协人大': 22, '自媒体': 23

利用[哈工大讯飞联合实验室‘RoBERTa-wwm-ext-large’](https://github.com/ymcui/Chinese-BERT-wwm)模型。

- 实验发现对文本处理时统一长度对效果有明显的改善，在线性空间的视角下，词向量在同一维度下表示同一个文本属性，可能更利于分类器对空间的分割，从而获得了更好的效果。

## 内容导引

|                             章节                             |            描述            |
| :----------------------------------------------------------: | :------------------------: |
| [数据分析](https://github.com/hawkforever5/BERT_User-Classification#对于数据的分析) |     对标记数据进行分析     |
| [工具包](https://github.com/hawkforever5/BERT_User-Classification#实验步骤) |         自制工具包         |
| [实验步骤](https://github.com/hawkforever5/BERT_User-Classification#实验步骤) | 简述实验过程并展示实验结果 |
|                                                              |                            |

## 

## 数据分析

- 标记数据1.2w条，由于数据量较少，适合采用BERT模型进行微调的策略。

- 剔除极端数据('明星红人', '民主党派')共有22类。

- 可利用信息包括昵称、关注数、粉丝数、微博数、认证信息、博主标记、简介、工作信息、标签和其他，其中3项为整型数据，6项为字符型数据。

# 工具包

待完善




## 实验步骤

### 一、模型未涉及数据

- | 明星红人 | 民主党派 |
  | :------: | :------: |

  两类标签对应的数据过少，不适宜放入模型训练，故需用规则剔除。

### 二、训练集规模(size_of_data)

- 训练集的各标签的数据规模不统一，经实验发现，这在反向传播的过程中会误导模型对参数的调整，导致模型只追求整体分类效果而忽略各标签的分类效果，故应以各标签的数据量为依据，将数据集分为大小数据集分开训练，实验证明，此方法对效果有明显的改善。

大数据： 

| **社区组织** |     **党委**     |   **自媒体**   |  **网民**   |   **媒体**   | **司法机关** |
| :----------: | :--------------: | :------------: | :---------: | :----------: | :----------: |
|   **学校**   | **超话粉丝大咖** | **企事业单位** | **大V名人** | **社会组织** |   **政府**   |

小数据：

|   **基层组织**   | **赛事活动** | **研究机构** |   **检验检测**   | **政协人大** |
| :--------------: | :----------: | :----------: | :--------------: | :----------: |
| **演艺娱乐明星** |  **公务员**  | **行业专家** | **外国政府机构** | **国防军委** |

实验结果：

![result1. size_of_data.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result1.%20size_of_data.png?raw=true)

{'大数据': 0, '小数据': 1}

### 三、小数据(small_data_10)

修改 Config字典：'model_save_path': 'small_data_10.pth'

分出小数据后，直接进行10分类。

实验结果：

![result2. small_data_10.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result2.%20small_data_10.png?raw=true)

{'公务员': 0, '国防军委': 1, '基层组织': 2, '检验检测': 3, '赛事活动': 4, '外国政府机构': 5, '行业专家': 6, '研究机构': 7, '演艺娱乐明星': 8, '政协人大': 9}

### 四、大数据(big_data_12)

尝试直接对大数据进行12分类。

实验结果：

![result3. big_data_12.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result3.%20big_data_12.png?raw=true)

效果并不理想。{'超话粉丝大咖': 0, '大V名人': 1, '党委': 2, '政府': 3, '媒体': 4, '企事业单位': 5, '社会组织': 6, '社区组织': 7, '司法机关': 8, '网民': 9, '学校': 10, '自媒体': 11}

### 五、数据在特征空间的离散程度(similar_distinctive)

- 大数据类中，部分标签所对应数据在特征空间中具有明显的区分度，向量内积优秀，故可直接进行n+1分类将其解决。

  实验结果：

![result4. similar_distinctive.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result4.%20similar_distinctive.png?raw=true)

{'distinctive': 0, 'similar': 1}


### 六、高区分度数据（distinctive_data）

修改 Config字典：'model_save_path': 'distinctive_data.pth'

实验结果：

![result5.  distinctive_data.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result5.%20%20distinctive_data.png?raw=true)

{'超话粉丝大咖': 0, '政府': 1, '社会组织': 2, '社区组织': 3, '司法机关': 4, '学校': 5}

### 七、低区分度数据(similar_6)

![result6.  similar_6.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result6.%20%20similar_6.png?raw=true)

{'大V名人': 0, '党委': 1, '媒体': 2, '企事业单位': 3, '网民': 4, '自媒体': 5}

### 八、意外发现 (similar-first)

- 在对将党委加入distinctive_data实验时，发现效果并不好，在对剩余数据进行6分类时，党委的分类准确率远高于其他，验证过后，决定在此单独进行分类。

分出党委

![result7.  similar-first.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result7.%20%20similar-first.png?raw=true)

{'其他': 0, '党委': 1}

### 九、影响准确率的关键因素(netzen+we-other)

分出网友+自媒体

![result8. netzen+we-other.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result8.%20netzen+we-other.png?raw=true)

{'其他': 0, '分出网友+自媒体': 1}

### 十、收尾数据(last3)

![result9. last3.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result9.%20last3.png?raw=true)

{'大V名人': 0, '媒体': 1, '企事业单位': 2}

