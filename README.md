# 微博发声人鉴别

[**中文说明**](https://github.com/hawkforever5/BERT_User-Classification/blob/main/README.md) | [**English**](https://github.com/hawkforever5/BERT_User-Classification/blob/main/README_EN.md)

利用微博的用户数据对用户进行分类。

初步共有24个分类：'超话粉丝大咖': 0, '公务员': 1, '大V名人': 2, '党委': 3, '国防军委': 4, '基层组织': 5, '政府': 6, '检验检测': 7, '媒体': 8, '民主党派': 9, '明星红人': 10, '企事业单位': 11, '赛事活动': 12, '社会组织': 13, '社区组织': 14, '司法机关': 15, '外国政府机构': 16, '网民': 17, '行业专家': 18, '学校': 19, '研究机构': 20, '演艺娱乐明星': 21, '政协人大': 22, '自媒体': 23

利用[哈工大讯飞联合实验室‘RoBERTa-wwm-ext-large’](https://github.com/ymcui/Chinese-BERT-wwm)模型。

- 实验发现对文本处理时统一长度对效果有明显的改善，在线性空间的视角下，词向量在同一维度下表示同一个文本属性，可能更利于分类器对空间的分割，从而获得了更好的效果。
