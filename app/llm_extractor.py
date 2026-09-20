"""LLM 知识抽取模块。

- 有 DeepSeek Key  → 真实抽取（强制 JSON，低温度，失败重试）
- 无 Key           → 自动降级为 MockExtractor，用轻量关键词规则从文档抽取，
                     保证"没有 key 也能演示完整闭环"。
- 大文档           → 按章节分段抽取后合并，避免截断丢失内容。
"""
import json
import re
from typing import List

from .prompts import KNOWLEDGE_EXTRACTION_SYSTEM, truncate
from .schemas import ExtractionResult, KnowledgeEntity, KnowledgeRelation

try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:  # pragma: no cover
    _HAS_OPENAI = False

MAX_CHUNK_CHARS = 5000


def _dedup(entities: List[KnowledgeEntity], relations: List[KnowledgeRelation]) -> ExtractionResult:
    """按实体名去重，并过滤掉两端实体的引用不存在的边。"""
    names = set()
    clean_entities = []
    for ent in entities:
        key = ent.name.strip()
        if not key or key in names:
            continue
        names.add(key)
        clean_entities.append(ent)
    clean_rels = []
    for rel in relations:
        if rel.source in names and rel.target in names and rel.type:
            clean_rels.append(rel)
    return ExtractionResult(entities=clean_entities, relations=clean_rels)


def _merge_results(results: List[ExtractionResult]) -> ExtractionResult:
    """合并多段抽取结果并去重。"""
    all_entities = []
    all_relations = []
    for r in results:
        all_entities.extend(r.entities)
        all_relations.extend(r.relations)
    return _dedup(all_entities, all_relations)


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """按章节/段落边界分段，避免在句子中间截断。"""
    if len(text) <= max_chars:
        return [text]

    # 尝试按章节标题分割（# 标题 或 第X章 等）
    chapter_pattern = re.compile(r'^(?:#{1,3}\s|第[一二三四五六七八九十\d]+[章节])', re.MULTILINE)
    chapters = chapter_pattern.split(text)

    chunks = []
    for chapter in chapters:
        chapter = chapter.strip()
        if not chapter:
            continue
        if len(chapter) <= max_chars:
            chunks.append(chapter)
        else:
            # 章节仍过长，按段落进一步分割
            paragraphs = chapter.split('\n\n')
            current = ''
            for para in paragraphs:
                if len(current) + len(para) + 2 <= max_chars:
                    current = current + '\n\n' + para if current else para
                else:
                    if current:
                        chunks.append(current)
                    current = para
            if current:
                chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


class DeepSeekExtractor:
    """真实大模型抽取，按 OpenAI 协议访问 DeepSeek。

    大文档自动分段抽取后合并。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        if not _HAS_OPENAI:
            raise RuntimeError("请先安装 openai：pip install openai")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.name = "deepseek"

    def extract(self, text: str) -> ExtractionResult:
        try:
            chunks = split_into_chunks(text)
            if len(chunks) == 1:
                return self._extract_single(chunks[0])
            results = [self._extract_single(chunk) for chunk in chunks]
            return _merge_results(results)
        except Exception as e:  # 余额不足/网络故障等：回退 mock，避免接口中断
            print(f"[llm_extractor] 真实抽取失败，已降级 mock：{type(e).__name__}: {e}", flush=True)
            if not hasattr(self, "_fallback"):
                self._fallback = MockExtractor()
            return self._fallback.extract(text)

    def _extract_single(self, text: str) -> ExtractionResult:
        user_prompt = truncate(text)
        messages = [
            {"role": "system", "content": KNOWLEDGE_EXTRACTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
        entities = [KnowledgeEntity(**e) for e in data.get("entities", [])]
        relations = [KnowledgeRelation(**r) for r in data.get("relations", [])]
        return _dedup(entities, relations)


# ---------- 模拟抽取（无 Key 降级）----------
# 预置知识库：超过 40 个常见机器学习 + 深度学习知识点。mock 抽取时，文档里出现哪个名字就抽出哪个。
_DEMO_ENTITIES = {
    # ---- 机器学习 ----
    "机器学习": ("概念", "从数据中学习规律并用于预测或决策"),
    "监督学习": ("方法", "使用带标签数据训练模型"),
    "无监督学习": ("方法", "处理无标签数据，常通过聚类发现结构"),
    "强化学习": ("方法", "通过与环境交互改善策略"),
    "特征": ("概念", "对原始数据的数值化描述"),
    "线性回归": ("算法", "学习输入特征到连续输出之间线性关系的算法"),
    "逻辑回归": ("算法", "常用于二分类问题的算法"),
    "决策树": ("算法", "通过一系列特征判断来划分数据的算法"),
    "K均值": ("算法", "最具代表性的聚类算法"),
    "过拟合": ("概念", "模型在训练集表现好但泛化能力差的现象"),
    "准确率": ("指标", "分类任务中预测正确的比例"),
    "泛化能力": ("概念", "模型在未见数据上的表现能力"),
    "偏差方差权衡": ("概念", "模型偏差与方差之间的权衡关系"),
    "正则化": ("方法", "通过惩罚项抑制过拟合、简化模型"),
    "交叉验证": ("方法", "将数据切分为多份轮流训练验证的评估方法"),
    "训练集": ("概念", "用于训练模型的数据子集"),
    "验证集": ("概念", "用于调参选择模型的数据子集"),
    "测试集": ("概念", "用于评估最终模型泛化能力的数据子集"),
    "梯度下降": ("算法", "通过沿负梯度方向迭代更新参数的优化方法"),
    "学习率": ("概念", "梯度下降每次更新的步长"),
    "朴素贝叶斯": ("算法", "基于贝叶斯定理的生成式分类算法"),
    "支持向量机": ("算法", "寻找最大间隔超平面进行分类的算法"),
    "随机森林": ("算法", "由多棵决策树集成投票的算法"),
    "集成学习": ("方法", "组合多个模型以提升整体性能"),
    "K 近邻": ("算法", "根据最近邻样本的类别进行判断的算法"),
    "特征工程": ("方法", "构造、选取和变换特征以提升模型效果"),
    "样本": ("概念", "数据集中的一条实例"),
    "标签": ("概念", "样本对应的目标输出值"),
    "分类问题": ("概念", "预测离散类别的监督学习任务"),
    "回归问题": ("概念", "预测连续数值的监督学习任务"),
    "数据预处理": ("方法", "对原始数据做清洗、标准化等处理"),
    "超参数": ("概念", "在训练前人工设定的模型参数"),
    "损失函数": ("概念", "衡量模型预测与真实值差距的函数"),
    # ---- 深度学习 ----
    "深度学习": ("概念", "基于多层神经网络进行表示学习的方法"),
    "神经网络": ("概念", "由多层人工神经元堆叠组成的模型"),
    "人工神经元": ("概念", "神经网络的基本计算单元"),
    "激活函数": ("概念", "为神经元引入非线性的函数"),
    "前向传播": ("方法", "输入按层逐层向后计算得到输出"),
    "反向传播": ("方法", "按链式法则从输出端向前计算梯度"),
    "多层感知机": ("算法", "含多个隐藏层的基本前馈神经网络"),
    "卷积神经网络": ("算法", "利用卷积核提取局部特征的网络"),
    "循环神经网络": ("算法", "擅长处理序列数据的网络"),
    "长短期记忆网络": ("算法", "缓解长序列梯度消失的循环网络"),
    "卷积层": ("概念", "通过卷积核做特征提取的网络层"),
    "池化层": ("概念", "对特征图做下采样降低维度的网络层"),
    "全连接层": ("概念", "逐层全连接的分类输出层"),
    "Dropout": ("方法", "训练时随机丢弃部分神经元以抑制过拟合"),
    "批量归一化": ("方法", "对层输入做标准化以加速训练"),
    "优化器": ("概念", "负责更新模型参数的算法"),
    "迁移学习": ("方法", "将预训练模型知识迁移到新任务"),
    "注意力机制": ("概念", "让模型动态关注输入的关键部分"),
    "Transformer": ("算法", "基于自注意力机制的序列模型"),
    "词嵌入": ("概念", "把词映射为稠密向量的表示"),
    # ---- 机器学习各子课聚焦补充 ----
    "半监督学习": ("方法", "介于监督与无监督之间，混合使用两类数据"),
    "聚类": ("方法", "将相似样本自动归组的过程"),
    "主成分分析": ("算法", "通过正交变换提取主要成分实现降维"),
    "层次聚类": ("算法", "通过树状结构逐层合并样本的聚类方法"),
    "DBSCAN": ("算法", "基于密度的聚类算法"),
    "精确率": ("指标", "预测为正类样本中真实为正类的比例"),
    "召回率": ("指标", "真实正类中被正确找出的比例"),
    "F1分数": ("指标", "精确率与召回率的调和平均"),
    "混淆矩阵": ("概念", "统计各类别预测结果分布的方阵"),
    "ROC曲线": ("概念", "反映不同分类阈值下误报率与召回率关系的曲线"),
    "AUC": ("指标", "ROC 曲线下面积，衡量二分类整体性能"),
    # ---- 深度学习各子课聚焦补充 ----
    "门控循环单元": ("算法", "带门控机制的简化循环网络"),
    "序列到序列": ("算法", "将输入序列映射为输出序列的模型结构"),
    "梯度消失": ("概念", "深层网络反向传播时梯度趋近于零的现象"),
    "自注意力": ("概念", "序列内部元素之间互相加权关注"),
    "多头注意力": ("方法", "并行多个注意力头以捕获不同特征"),
    "位置编码": ("概念", "为序列叠加位置信息的编码"),
    "预训练": ("方法", "在大规模语料上预先训练模型"),
    "微调": ("方法", "在目标任务上对预训练模型做再训练"),
}
_DEMO_RELATIONS = [
    ("机器学习", "监督学习", "has_part"),
    ("机器学习", "无监督学习", "has_part"),
    ("机器学习", "强化学习", "has_part"),
    ("深度学习", "机器学习", "specializes"),
    ("监督学习", "线性回归", "has_part"),
    ("监督学习", "逻辑回归", "has_part"),
    ("监督学习", "决策树", "has_part"),
    ("监督学习", "朴素贝叶斯", "has_part"),
    ("监督学习", "支持向量机", "has_part"),
    ("监督学习", "K 近邻", "has_part"),
    ("监督学习", "随机森林", "has_part"),
    ("无监督学习", "K均值", "has_part"),
    ("集成学习", "随机森林", "has_part"),
    ("过拟合", "正则化", "reduces"),
    ("过拟合", "交叉验证", "related"),
    ("监督学习", "训练集", "uses"),
    ("监督学习", "验证集", "uses"),
    ("监督学习", "测试集", "uses"),
    ("特征", "特征工程", "related"),
    ("机器学习", "泛化能力", "related"),
    ("偏差方差权衡", "过拟合", "related"),
    ("机器学习", "数据预处理", "uses"),
    ("机器学习", "分类问题", "related"),
    ("机器学习", "回归问题", "related"),
    ("逻辑回归", "损失函数", "uses"),
    ("线性回归", "梯度下降", "related"),
    ("梯度下降", "学习率", "related"),
    ("决策树", "过拟合", "related"),
    ("准确率", "分类问题", "related"),
    ("深度学习", "神经网络", "has_part"),
    ("深度学习", "卷积神经网络", "has_part"),
    ("深度学习", "循环神经网络", "has_part"),
    ("深度学习", "多层感知机", "has_part"),
    ("深度学习", "Transformer", "has_part"),
    ("神经网络", "人工神经元", "has_part"),
    ("神经网络", "激活函数", "uses"),
    ("神经网络", "前向传播", "uses"),
    ("神经网络", "反向传播", "uses"),
    ("神经网络", "损失函数", "uses"),
    ("神经网络", "优化器", "uses"),
    ("神经网络", "Dropout", "uses"),
    ("神经网络", "批量归一化", "uses"),
    ("卷积神经网络", "卷积层", "has_part"),
    ("卷积神经网络", "池化层", "has_part"),
    ("卷积神经网络", "全连接层", "has_part"),
    ("循环神经网络", "长短期记忆网络", "specializes"),
    ("反向传播", "梯度下降", "uses"),
    ("Transformer", "注意力机制", "uses"),
    ("深度学习", "迁移学习", "has_part"),
    ("深度学习", "词嵌入", "uses"),
    # ---- 机器学习各子课聚焦关系 ----
    ("机器学习", "半监督学习", "has_part"),
    ("无监督学习", "主成分分析", "has_part"),
    ("无监督学习", "层次聚类", "has_part"),
    ("无监督学习", "DBSCAN", "has_part"),
    ("K均值", "DBSCAN", "contrast"),
    ("聚类", "主成分分析", "related"),
    ("机器学习", "混淆矩阵", "related"),
    ("分类问题", "精确率", "related"),
    ("分类问题", "召回率", "related"),
    ("分类问题", "F1分数", "related"),
    ("分类问题", "混淆矩阵", "related"),
    ("分类问题", "ROC曲线", "related"),
    ("分类问题", "AUC", "related"),
    ("准确率", "F1分数", "related"),
    # ---- 深度学习各子课聚焦关系 ----
    ("深度学习", "预训练", "uses"),
    ("预训练", "微调", "leads_to"),
    ("循环神经网络", "门控循环单元", "specializes"),
    ("循环神经网络", "序列到序列", "has_part"),
    ("循环神经网络", "梯度消失", "related"),
    ("长短期记忆网络", "门控循环单元", "contrast"),
    ("长短期记忆网络", "梯度消失", "reduces"),
    ("Transformer", "自注意力", "uses"),
    ("Transformer", "多头注意力", "uses"),
    ("Transformer", "位置编码", "uses"),
    ("自注意力", "注意力机制", "has_part"),
]


class MockExtractor:
    """演示用：把文档中出现的预置知识点筛选出来，返回一个真实依赖文本的图谱。

    便于你未配置 API Key 时：先跑通上传→抽取→建图→展示整条闭环。
    """

    name = "mock"

    def extract(self, text: str) -> ExtractionResult:
        present = [name for name in _DEMO_ENTITIES if name in text]
        entities = [
            KnowledgeEntity(name=n, kind=k, definition=d)
            for n in present
            for (k, d) in [_DEMO_ENTITIES[n]]
        ]
        relations = [
            KnowledgeRelation(source=s, target=t, type=rtype)
            for (s, t, rtype) in _DEMO_RELATIONS
            if s in present and t in present
        ]
        return ExtractionResult(entities=entities, relations=relations)


def build_extractor():
    """按配置返回真实大模型抽取器（豆包 / DeepSeek），未配置 Key 时降级为 mock。"""
    from . import config
    if config.use_mock_llm():
        return MockExtractor()
    if config.LLM_PROVIDER == "doubao":
        ex = DeepSeekExtractor(config.DOUBAO_API_KEY, config.DOUBAO_BASE_URL, config.DOUBAO_MODEL)
        ex.name = "doubao"
        return ex
    return DeepSeekExtractor(config.LLM_API_KEY, config.LLM_BASE_URL, config.LLM_MODEL)
