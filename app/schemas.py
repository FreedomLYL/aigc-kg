"""Pydantic 数据模型：结构上与前端 / 图谱存储对齐。"""
from typing import List, Optional
from pydantic import BaseModel, Field


class KnowledgeEntity(BaseModel):
    name: str = Field(..., description="知识点名称，如：线性回归")
    kind: str = Field(default="概念", description="实体类型：概念/公式/定理/算法/术语 等")
    definition: Optional[str] = Field(default=None, description="一句话定义")


class KnowledgeRelation(BaseModel):
    source: str = Field(..., description="源知识点名")
    target: str = Field(..., description="目标知识点名")
    type: str = Field(..., description="关系类型：前置/包含/相关/应用 等")


class ExtractionResult(BaseModel):
    entities: List[KnowledgeEntity] = []
    relations: List[KnowledgeRelation] = []


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色：user / assistant")
    content: str = Field(..., description="消息内容")


class RequestChat(BaseModel):
    question: str = Field(..., min_length=1)
    history: List[ChatMessage] = Field(default_factory=list, description="多轮对话历史（上文的 user/assistant 消息）")
    provider: str = Field("deepseek", description="模型源：doubao / deepseek（页面内直接使用）")


class RecommendRequest(BaseModel):
    mastered: List[str] = Field(default_factory=list, description="已掌握的知识点集合")
    top_k: int = Field(default=5, ge=1, le=20)


class NodeCreate(BaseModel):
    name: str = Field(..., description="知识点名称")
    kind: str = Field(default="概念", description="实体类型")
    definition: Optional[str] = Field(default=None, description="一句话定义")


class NodeUpdate(BaseModel):
    name: Optional[str] = Field(default=None, description="新名称（改名时填写）")
    kind: Optional[str] = Field(default=None, description="新类型")
    definition: Optional[str] = Field(default=None, description="新定义")


class EdgeCreate(BaseModel):
    source: str = Field(..., description="源知识点名")
    target: str = Field(..., description="目标知识点名")
    type: str = Field(..., description="关系类型：pre/has_part/related/applies_to/contrast")


class EdgeDelete(BaseModel):
    source: str = Field(...)
    target: str = Field(...)
    type: Optional[str] = Field(default=None, description="指定类型则只删该关系，否则删所有")


# ---------- 认证 ----------

class CourseCreate(BaseModel):
    course_id: str = Field(..., min_length=1, max_length=64, description="课程名称（不能包含数字）")


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=32, description="用户名")
    password: str = Field(..., min_length=1, max_length=64, description="密码")
    role: str = Field("student", description="身份：student / teacher")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class AssistantRequest(BaseModel):
    course_id: str = Field(..., description="课程 ID")
    tool_id: str = Field("deepseek", description="AI 工具 id，如 doubao/feishu/workbuddy/trae/ima/deepseek")
    task: str = Field("", max_length=500, description="教学任务描述，如：为「过拟合」生成 3 道测验题")


# ---------- 课后习题 / 作业布置 / 学情 / 教案 ----------

class HomeworkAssignRequest(BaseModel):
    course_id: str = Field(..., description="课程 ID")
    n: int = Field(default=5, ge=1, le=50, description="题数")
    requirement: str = Field("", max_length=500, description="布置要求（可选）")


class HomeworkClearRequest(BaseModel):
    course_id: str = Field(..., description="课程 ID")


class QuizSubmitRequest(BaseModel):
    course_id: str = Field(..., description="课程 ID")
    total: int = Field(default=0, ge=0, description="本组题数")
    correct: int = Field(default=0, ge=0, description="答对题数")
    wrong_points: List[str] = Field(default_factory=list, description="答错的知识点")


class QuizVariantRequest(BaseModel):
    course_id: str = Field(..., description="课程 ID")
    point: str = Field(..., description="需要巩固的薄弱知识点")
    n: int = Field(default=3, ge=1, le=10, description="变式题数")


class TeachingPlanRequest(BaseModel):
    course_id: str = Field(..., description="课程 ID")