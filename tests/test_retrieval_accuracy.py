"""检索召回准确率测试 — 使用真实 Embedding 模型，构造大量多样化记忆数据，验证检索命中率。
运行方式: python -m pytest tests/test_retrieval_accuracy.py -v -s
"""
from __future__ import annotations

import sys
import time
import uuid

sys.path.insert(0, "/Users/maidong/Desktop/zyc/研究openclaw/agent-memory")

import numpy as np

from memory_agent.config import settings
from memory_agent.memory.search import MemorySearcher
from memory_agent.providers.embedding_local import LocalEmbeddingProvider
from memory_agent.store.sqlite import SQLiteMemoryStore
from memory_agent.types import MemoryRecord, MemoryType

# ══════════════════════════════════════════════════════════
# 测试数据：模拟真实用户的完整画像
# ══════════════════════════════════════════════════════════

# 每条记忆: (type, name, description, content, importance)
SEED_MEMORIES: list[tuple[MemoryType, str, str, str, float]] = [
    # ═══════════════════════════════════════════
    # 用户画像（USER）— 约 50 条
    # ═══════════════════════════════════════════
    (MemoryType.USER, "姓名赵云", "用户真名赵云，昵称小赵", "用户名叫赵云，昵称小赵，习惯别人叫他小赵", 0.9),
    (MemoryType.USER, "广州居住", "用户住在广州天河区", "用户居住在广州市天河区，之前在深圳待过两年", 0.8),
    (MemoryType.USER, "产品开发", "用户是企业产品开发人员", "用户职业是企业产品开发人员，负责AI方向的产品设计和开发", 0.9),
    (MemoryType.USER, "Python技术栈", "用户主力语言Python，熟悉FastAPI", "用户技术栈以 Python 为主，熟悉 FastAPI、SQLite、向量检索，正在学习 Rust", 0.8),
    (MemoryType.USER, "喜欢火锅", "用户爱吃火锅和粤菜", "用户喜欢吃火锅，尤其是四川火锅，也喜欢粤菜煲汤", 0.6),
    (MemoryType.USER, "周杰伦粉丝", "用户喜欢听周杰伦的歌", "用户是周杰伦粉丝，最喜欢《晴天》和《稻香》，平时也听林俊杰", 0.6),
    (MemoryType.USER, "养猫", "用户养了一只橘猫叫咪咪", "用户家里养了一只橘猫，名字叫咪咪，已经三岁了", 0.7),
    (MemoryType.USER, "女友小美", "用户女朋友叫小美", "用户女朋友叫小美，是做UI设计的，两人在一起两年了", 0.7),
    (MemoryType.USER, "健身爱好", "用户每周去健身房三次", "用户有健身习惯，每周去健身房三次，主要练力量训练", 0.5),
    (MemoryType.USER, "近视400度", "用户近视400度戴眼镜", "用户近视400度，一直戴框架眼镜，考虑做近视手术", 0.4),
    (MemoryType.USER, "过敏体质", "用户对海鲜过敏", "用户对海鲜过敏，尤其是虾和螃蟹，吃了会起疹子", 0.6),
    (MemoryType.USER, "失眠问题", "用户最近经常失眠", "用户最近工作压力大，经常失眠到凌晨两三点才能睡着", 0.5),
    (MemoryType.USER, "喜欢打篮球", "周末爱打篮球", "用户周末经常和同事约着打篮球，最喜欢的NBA球星是库里", 0.4),
    (MemoryType.USER, "不喝酒", "用户不喝酒", "用户不喝酒，喝一点就脸红，聚餐时一般喝果汁或茶", 0.4),
    (MemoryType.USER, "身高175", "用户身高175cm", "用户身高175cm，体重70kg，BMI正常", 0.3),
    (MemoryType.USER, "生日九月", "用户生日是9月15日", "用户生日是9月15日，处女座", 0.5),
    (MemoryType.USER, "老家湖南", "用户老家湖南长沙", "用户老家在湖南长沙，每年春节回去过年", 0.5),
    (MemoryType.USER, "弟弟在读大学", "用户有个弟弟在读大学", "用户有个弟弟叫赵风，今年大三，在武汉大学读计算机", 0.5),
    (MemoryType.USER, "父亲退休", "用户父亲已退休", "用户父亲今年60岁，去年从湖南某中学退休，以前是数学老师", 0.4),
    (MemoryType.USER, "母亲做菜好", "用户母亲厨艺好", "用户母亲做菜很好吃，尤其擅长做湖南菜，辣椒炒肉是拿手菜", 0.4),
    (MemoryType.USER, "喜欢看科幻电影", "用户爱看科幻片", "用户喜欢看科幻电影，最喜欢《星际穿越》和《流浪地球》", 0.4),
    (MemoryType.USER, "玩原神", "用户平时玩原神", "用户休息时间会玩原神，主玩钟离和雷电将军", 0.3),
    (MemoryType.USER, "喝美式咖啡", "用户每天喝美式咖啡", "用户每天早上都要喝一杯美式咖啡，不加糖不加奶", 0.4),
    (MemoryType.USER, "开本田思域", "用户开本田思域", "用户开一辆白色本田思域，2022款，平时通勤用", 0.4),
    (MemoryType.USER, "微信号xiaozhao", "用户微信号", "用户微信号是 xiaozhao_gz，头像是橘猫咪咪", 0.3),
    (MemoryType.USER, "喜欢极简风格", "用户偏好极简设计", "用户家里装修选了极简风格，喜欢黑白灰配色，不喜欢花哨的装饰", 0.4),
    (MemoryType.USER, "用iPhone", "用户用iPhone 15 Pro", "用户手机是 iPhone 15 Pro，256GB，之前用华为", 0.3),
    (MemoryType.USER, "喜欢雨天", "用户喜欢下雨天", "用户说他很喜欢下雨天，觉得雨声很治愈，适合写代码", 0.3),
    (MemoryType.USER, "不吃香菜", "用户不吃香菜", "用户不吃香菜，闻到味道就受不了，点外卖总是备注不要香菜", 0.4),
    (MemoryType.USER, "有花粉过敏", "春天花粉过敏", "用户春天会花粉过敏，出门要戴口罩，严重时需要吃抗过敏药", 0.4),
    (MemoryType.USER, "本科华工", "用户本科华南理工", "用户本科毕业于华南理工大学软件工程专业，2018年毕业", 0.5),

    # ═══════════════════════════════════════════
    # 项目上下文（PROJECT）— 约 60 条
    # ═══════════════════════════════════════════
    (MemoryType.PROJECT, "记忆系统开发", "正在开发基于SQLite的三层记忆系统", "用户正在开发一个三层记忆系统，基于 SQLite + 向量检索 + BM25 混合搜索", 0.9),
    (MemoryType.PROJECT, "OpenClaw研究", "在研究OpenClaw开源项目", "用户在研究 OpenClaw 开源项目，想从中提取设计模式融入企业产品", 0.8),
    (MemoryType.PROJECT, "周五演示", "本周五要给领导做产品演示", "用户本周五需要给领导做产品演示，展示记忆系统的核心功能", 0.9),
    (MemoryType.PROJECT, "部署方案", "计划用Docker部署到公司服务器", "记忆系统计划用 Docker 部署到公司内网服务器，需要支持 GPU 加速", 0.7),
    (MemoryType.PROJECT, "性能优化", "ONNX推理性能优化中", "用户正在将 embedding 模型从 PyTorch 迁移到 ONNX Runtime，已实现 66% 加速", 0.7),
    (MemoryType.PROJECT, "多用户支持", "计划支持多用户隔离", "下一步计划是支持多用户隔离，每个用户有独立的记忆空间", 0.6),
    (MemoryType.PROJECT, "珠海长隆", "计划周末去珠海长隆玩", "用户计划这个周末和女友小美一起去珠海长隆海洋王国玩，2大1小的票", 0.6),
    (MemoryType.PROJECT, "买新电脑", "考虑买MacBook Pro M4", "用户在考虑换电脑，想买 MacBook Pro M4 Max 版本，预算三万左右", 0.5),
    (MemoryType.PROJECT, "学车中", "用户在学驾照科目二", "用户正在驾校学车，目前在练科目二，觉得倒车入库很难", 0.5),
    (MemoryType.PROJECT, "读书计划", "在读《设计模式》", "用户最近在读《设计模式：可复用面向对象软件的基础》，读到第三章了", 0.4),
    (MemoryType.PROJECT, "装修房子", "用户新房在装修", "用户在天河区买了套新房，正在装修中，预计下个月完工", 0.5),
    (MemoryType.PROJECT, "年假计划", "国庆去日本旅游", "用户计划国庆假期和女友小美一起去日本东京旅游，已经在看机票了", 0.5),
    (MemoryType.PROJECT, "FTS5分词优化", "FTS5全文检索分词器改进", "正在优化 FTS5 的中文分词效果，考虑引入 jieba 分词替代逐字切分", 0.6),
    (MemoryType.PROJECT, "向量维度选择", "Embedding模型384维", "选定 paraphrase-multilingual-MiniLM-L12-v2 作为 embedding 模型，输出 384 维", 0.5),
    (MemoryType.PROJECT, "Claude集成", "通过CLI方式集成Claude", "记忆系统通过 Claude CLI subprocess 方式接入 LLM，支持 Sonnet 和 Haiku", 0.6),
    (MemoryType.PROJECT, "Web界面开发", "React前端界面开发中", "正在开发 React 前端界面，用于可视化展示记忆数据和搜索结果", 0.5),
    (MemoryType.PROJECT, "团队分享准备", "下周二要做技术分享", "用户下周二要在部门内做一次技术分享，主题是向量检索在记忆系统中的应用", 0.7),
    (MemoryType.PROJECT, "压力测试计划", "需要做并发压力测试", "计划对记忆系统做并发压力测试，验证 SQLite WAL 模式下的写入吞吐量", 0.5),
    (MemoryType.PROJECT, "数据迁移脚本", "写旧数据迁移脚本", "需要写一个数据迁移脚本，把旧版记忆数据迁移到新的 schema", 0.5),
    (MemoryType.PROJECT, "API接口设计", "RESTful API接口设计", "记忆系统对外暴露 RESTful API，端口 8899，支持 JSON 格式", 0.5),
    (MemoryType.PROJECT, "咪咪疫苗", "猫咪咪要打疫苗", "咪咪下个月要打三联疫苗，已经预约了天河区宠物医院", 0.4),
    (MemoryType.PROJECT, "考虑买投影仪", "考虑给新房买投影仪", "装修完想给客厅买个投影仪，在看极米和当贝，预算五千", 0.3),
    (MemoryType.PROJECT, "准备求婚", "用户打算明年向小美求婚", "用户打算明年情人节向小美求婚，在偷偷看钻戒，预算两万", 0.6),
    (MemoryType.PROJECT, "学Rust", "正在学习Rust语言", "用户开始学习 Rust 语言，在看《Rust程序设计语言》官方教程，觉得所有权概念很难", 0.5),
    (MemoryType.PROJECT, "日语学习", "在用多邻国学日语", "用户在用多邻国学日语，为国庆去日本旅游做准备，目前学到N5水平", 0.4),
    (MemoryType.PROJECT, "减脂计划", "最近在控制饮食减脂", "用户最近在减脂，控制碳水摄入，晚餐只吃鸡胸肉和蔬菜", 0.4),
    (MemoryType.PROJECT, "牙齿矫正", "在做隐适美矫正", "用户在做隐适美牙齿矫正，已经戴了8个月，还剩4个月", 0.4),
    (MemoryType.PROJECT, "换发型", "最近想换个发型", "用户想把头发剪短，从中长发换成寸头，还在犹豫", 0.2),
    (MemoryType.PROJECT, "驾照科目三", "科目二过了在练科目三", "用户科目二上周通过了，这周开始练科目三，觉得灯光模拟很复杂", 0.5),
    (MemoryType.PROJECT, "小美生日礼物", "在给小美选生日礼物", "小美下个月过生日，用户在考虑送一个 Dyson 吹风机或者一条项链", 0.5),
    (MemoryType.PROJECT, "搬家计划", "装修完准备搬家", "新房装修完后计划下个月搬家，在联系搬家公司比价", 0.4),
    (MemoryType.PROJECT, "组装NAS", "想组装一台NAS", "用户想组装一台 NAS 存照片和电影，在研究群晖和 TrueNAS", 0.3),
    (MemoryType.PROJECT, "公司年会节目", "公司年会要表演节目", "公司年底年会，用户被拉去演小品，角色是程序员，正在背台词", 0.3),
    (MemoryType.PROJECT, "考虑跳槽", "在观望其他工作机会", "用户对当前薪资不太满意，在看其他公司的 AI 产品岗位机会", 0.5),
    (MemoryType.PROJECT, "做副业", "想做AI相关自媒体", "用户在考虑做 AI 技术方向的自媒体，打算先在知乎和B站写文章", 0.4),
    (MemoryType.PROJECT, "弟弟考研", "弟弟在准备考研", "用户弟弟赵风准备考研，目标是中科大计算机系，用户在帮他复习英语", 0.4),
    (MemoryType.PROJECT, "高中同学聚会", "下个月有高中同学聚会", "下个月有高中同学聚会在长沙，用户打算请两天假回去参加", 0.3),
    (MemoryType.PROJECT, "MacBook到货", "新MacBook已下单", "MacBook Pro M4 Max 已下单，预计下周到货，64GB内存版", 0.5),
    (MemoryType.PROJECT, "记忆系统优化改造", "参考Claude Code做优化改造", "参考 Claude Code 的记忆设计对记忆系统做了大规模优化改造，包括类型分类、线程安全、Token预算等", 0.8),

    # ═══════════════════════════════════════════
    # 行为指导（FEEDBACK）— 约 30 条
    # ═══════════════════════════════════════════
    (MemoryType.FEEDBACK, "用中文回答", "用户要求所有回答用中文", "用户要求所有回答必须用中文\n原因：用户英文阅读速度慢\n应用场景：所有对话场景", 0.9),
    (MemoryType.FEEDBACK, "简洁回复", "用户喜欢简洁的回答不要啰嗦", "用户不喜欢太长太啰嗦的回答，希望简洁直接\n原因：用户觉得冗长的回复浪费时间\n应用场景：日常对话和技术问答", 0.8),
    (MemoryType.FEEDBACK, "不要表情包", "用户不喜欢回复中带表情包", "不要在回复中使用 emoji 表情包\n原因：用户觉得显得不专业\n应用场景：所有回复", 0.7),
    (MemoryType.FEEDBACK, "代码要注释", "写代码时要加中文注释", "写代码时要加上简洁的中文注释\n原因：用户团队成员英文水平参差不齐\n应用场景：所有代码输出", 0.8),
    (MemoryType.FEEDBACK, "不要mock数据库", "测试中不要mock数据库", "写测试时不要 mock 数据库，使用真实数据库\n原因：上次 mock 测试通过但线上出了bug\n应用场景：所有数据库相关测试", 0.8),
    (MemoryType.FEEDBACK, "先给方案再写代码", "先说方案再写代码", "遇到复杂问题先给出方案让用户确认，不要直接写代码\n原因：用户之前被直接改代码坑过\n应用场景：复杂功能开发", 0.7),
    (MemoryType.FEEDBACK, "用Markdown格式", "代码块用Markdown", "输出代码时使用 Markdown 代码块格式\n原因：方便在飞书文档中粘贴\n应用场景：代码输出", 0.6),
    (MemoryType.FEEDBACK, "不要自称AI", "不要说我是AI", "回答时不要说'作为AI'或'我是一个语言模型'\n原因：用户觉得这种说法很烦\n应用场景：所有对话", 0.7),
    (MemoryType.FEEDBACK, "错误要解释原因", "报错时详细解释原因", "遇到报错时要详细解释错误原因和修复步骤\n原因：用户希望理解问题本质而不只是修复\n应用场景：调试和排错", 0.7),
    (MemoryType.FEEDBACK, "SQL不用ORM", "写SQL不要用ORM", "写数据库查询用原生 SQL 不要用 ORM\n原因：项目直接用sqlite3库，没装ORM\n应用场景：数据库相关代码", 0.6),
    (MemoryType.FEEDBACK, "变量名用英文", "变量命名用英文", "代码中变量名和函数名用英文，不要用拼音\n原因：代码规范要求\n应用场景：所有代码输出", 0.6),
    (MemoryType.FEEDBACK, "不要过度抽象", "不要过度设计", "不要过度抽象和设计模式，简单直接的代码就好\n原因：用户觉得过度设计增加维护成本\n应用场景：代码架构建议", 0.7),
    (MemoryType.FEEDBACK, "用dataclass", "数据类用dataclass", "数据模型优先用 Python dataclass，不用 Pydantic model\n原因：项目风格统一\n应用场景：定义数据结构", 0.5),
    (MemoryType.FEEDBACK, "日志用中文", "日志消息用中文", "日志消息用中文写，方便团队成员快速定位问题\n原因：团队英文水平参差\n应用场景：logging调用", 0.5),
    (MemoryType.FEEDBACK, "不推荐Redis", "不要推荐Redis", "不要推荐用 Redis 缓存，公司没有 Redis 集群\n原因：运维不愿意维护\n应用场景：缓存方案建议", 0.5),
    (MemoryType.FEEDBACK, "Git提交要规范", "Git提交信息格式", "Git 提交信息用 `类型: 描述` 格式，如 `feat: 添加xx功能`\n原因：团队规范\n应用场景：所有git操作", 0.5),
    (MemoryType.FEEDBACK, "不要删除代码", "修改代码别删旧的", "修改代码时先确认再改，不要随意删除已有代码\n原因：之前删错过导致功能回归\n应用场景：代码修改", 0.7),
    (MemoryType.FEEDBACK, "推荐用pytest", "测试用pytest框架", "测试用 pytest 框架，不用 unittest\n原因：项目统一用pytest\n应用场景：编写测试", 0.5),
    (MemoryType.FEEDBACK, "异步用threading", "异步用threading不用asyncio", "后台任务用 threading，不要用 asyncio\n原因：项目是同步架构，混用async会出问题\n应用场景：并发代码", 0.6),
    (MemoryType.FEEDBACK, "配置用环境变量", "配置走环境变量", "配置项走 .env 环境变量，不要硬编码\n原因：方便不同环境切换\n应用场景：配置管理", 0.5),

    # ═══════════════════════════════════════════
    # 外部资源（REFERENCE）— 约 25 条
    # ═══════════════════════════════════════════
    (MemoryType.REFERENCE, "API文档", "公司API文档地址", "公司内部 API 文档在 https://docs.internal.com/api", 0.6),
    (MemoryType.REFERENCE, "Linear看板", "Bug跟踪在Linear", "Bug 跟踪在 Linear 的 MEMORY 项目中，链接 https://linear.app/team/memory", 0.6),
    (MemoryType.REFERENCE, "Grafana监控", "系统监控看板在Grafana", "系统监控在 Grafana 看板 https://grafana.internal/d/memory-perf", 0.5),
    (MemoryType.REFERENCE, "飞书文档", "产品需求文档在飞书", "产品需求文档在飞书的「记忆系统PRD」文档中", 0.5),
    (MemoryType.REFERENCE, "GitHub仓库", "代码仓库在GitHub", "代码仓库地址 https://github.com/company/memory-agent", 0.5),
    (MemoryType.REFERENCE, "Claude文档", "Claude API官方文档", "Claude API 文档在 https://docs.anthropic.com", 0.5),
    (MemoryType.REFERENCE, "Sentry错误追踪", "错误追踪用Sentry", "线上错误追踪用 Sentry，项目名 memory-agent-prod", 0.5),
    (MemoryType.REFERENCE, "Jenkins构建", "CI/CD用Jenkins", "CI/CD 用公司内部 Jenkins，构建地址 https://jenkins.internal/job/memory-agent", 0.4),
    (MemoryType.REFERENCE, "Confluence文档", "架构设计文档在Confluence", "系统架构设计文档在 Confluence 的「AI产品组」空间下", 0.4),
    (MemoryType.REFERENCE, "钉钉告警群", "告警消息在钉钉群", "线上告警推送到钉钉机器人群「记忆系统告警」", 0.4),
    (MemoryType.REFERENCE, "npm私有仓库", "前端私有npm仓库", "前端组件库发布在公司私有 npm 仓库 https://npm.internal.com", 0.3),
    (MemoryType.REFERENCE, "sentence-transformers文档", "embedding模型文档", "embedding 模型文档 https://www.sbert.net/docs/pretrained_models.html", 0.4),
    (MemoryType.REFERENCE, "SQLite官方文档", "SQLite文档地址", "SQLite 官方文档 https://www.sqlite.org/docs.html，WAL模式说明在 https://www.sqlite.org/wal.html", 0.4),
    (MemoryType.REFERENCE, "宠物医院电话", "天河区宠物医院电话", "天河区芭比宠物医院电话：020-12345678，给咪咪看病的地方", 0.3),
    (MemoryType.REFERENCE, "健身房地址", "常去的健身房地址", "健身房是天河区体育西路的金吉鸟健身，会员到明年3月到期", 0.3),
    (MemoryType.REFERENCE, "ONNX Runtime文档", "ONNX Runtime官方文档", "ONNX Runtime 文档 https://onnxruntime.ai/docs/", 0.4),
    (MemoryType.REFERENCE, "FastAPI文档", "FastAPI官方文档", "FastAPI 官方文档 https://fastapi.tiangolo.com/", 0.4),
    (MemoryType.REFERENCE, "公司VPN", "公司VPN连接方式", "公司 VPN 用 WireGuard，配置文件在飞书IT支持文档里", 0.3),
    (MemoryType.REFERENCE, "装修公司联系", "装修公司联系方式", "装修公司是广州尚品宅配，项目经理姓陈，手机 138xxxxxxxx", 0.3),

    # ═══════════════════════════════════════════
    # 对话原文记录（模拟 extract 产生的低分记忆）— 约 40 条
    # 这些是较长的对话摘要，增加检索噪声难度
    # ═══════════════════════════════════════════
    (MemoryType.PROJECT, "讨论FTS分词", "讨论了FTS5分词方案", "用户和AI讨论了 FTS5 全文检索的中文分词方案，最终决定先用逐字切分加空格的简单方案", 0.4),
    (MemoryType.PROJECT, "讨论embedding模型", "对比了多个embedding模型", "用户和AI对比了 text2vec、bge-base、paraphrase-multilingual-MiniLM 三个模型，最终选了 MiniLM 因为多语言支持好且体积小", 0.4),
    (MemoryType.PROJECT, "调试reranker", "调试reranker精排功能", "用户在调试 bge-reranker-base 模型的 ONNX 版本，发现输出需要做 sigmoid 归一化", 0.4),
    (MemoryType.PROJECT, "讨论Pack压缩", "讨论Memory Pack压缩策略", "用户和AI讨论了 Memory Pack 的软边界检测策略，设计了时间间隔+信号词+关键词漂移三重检测", 0.4),
    (MemoryType.PROJECT, "重构store层", "重构了store抽象层", "用户把数据库操作抽象成 MemoryStore 接口，方便以后换 PostgreSQL 或其他存储", 0.4),
    (MemoryType.PROJECT, "配置热加载", "讨论配置热加载方案", "讨论了配置热加载方案，决定暂时用 pydantic-settings 的 .env 读取，不做热加载", 0.3),
    (MemoryType.PROJECT, "讨论WebSocket", "讨论了WebSocket推送", "讨论了 Web 前端是否需要 WebSocket 推送记忆更新，决定暂时用轮询方案", 0.3),
    (MemoryType.PROJECT, "设计CLI命令", "设计了CLI交互命令", "设计了 CLI 的 /core /memory /packs /stats 等交互命令，方便调试记忆数据", 0.3),
    (MemoryType.USER, "说起大学往事", "聊了大学趣事", "用户聊起大学时和室友一起参加ACM编程竞赛的经历，拿过省级银奖", 0.3),
    (MemoryType.USER, "吐槽广州天气", "吐槽广州天气热", "用户吐槽广州六月份太热了，每天出门就一身汗，想搬去昆明", 0.2),
    (MemoryType.USER, "讲咪咪趣事", "分享猫咪趣事", "用户分享说咪咪昨天偷吃了桌上的鱼，被小美追着满屋子跑", 0.2),
    (MemoryType.USER, "回忆小时候", "回忆长沙小时候", "用户回忆小时候在长沙吃臭豆腐和糖油粑粑，觉得现在广州买不到正宗的", 0.2),
    (MemoryType.PROJECT, "帮弟弟改论文", "帮弟弟改了毕业设计论文", "用户帮弟弟赵风改了毕业设计论文，主题是基于深度学习的图像分类", 0.3),
    (MemoryType.PROJECT, "修空调", "家里空调坏了要修", "用户说家里的格力空调不制冷了，约了售后明天上门维修", 0.2),
    (MemoryType.PROJECT, "咪咪挠沙发", "猫咪挠坏了沙发", "咪咪把客厅新沙发挠坏了一个角，用户在研究怎么防止猫挠家具", 0.2),
    (MemoryType.PROJECT, "约朋友烧烤", "周末约朋友户外烧烤", "用户计划这周日和大学同学在大学城约户外烧烤，需要买食材和工具", 0.3),
    (MemoryType.PROJECT, "健身受伤", "健身拉伤了腰", "用户上周硬拉时拉伤了腰，医生建议休息两周不要做重量训练", 0.4),
    (MemoryType.PROJECT, "研究RAG", "在研究RAG技术", "用户在研究 RAG（检索增强生成）技术，想把它融入记忆系统的检索流程", 0.5),
    (MemoryType.PROJECT, "给小美做饭", "学做红烧排骨", "用户在学做菜，最近学会了红烧排骨，打算小美生日那天做给她吃", 0.3),
    (MemoryType.PROJECT, "签证办理", "办日本旅游签证", "用户正在准备日本旅游签证材料，需要公司在职证明和银行流水", 0.4),

    # ═══════════════════════════════════════════
    # 补充记忆（凑到 200+ 条）
    # ═══════════════════════════════════════════
    # 更多工作细节
    (MemoryType.PROJECT, "重排序模型选型", "在对比reranker模型", "在对比 bge-reranker-base 和 cohere-reranker，最终选了 bge 因为可以本地部署", 0.5),
    (MemoryType.PROJECT, "写技术博客", "在写混合检索的技术博客", "用户在写一篇关于向量检索+BM25混合排序的技术博客，准备发在掘金", 0.4),
    (MemoryType.PROJECT, "对接语音识别", "对接火山引擎语音识别", "记忆系统在对接火山引擎的语音识别 API，实现语音转文字再提取记忆", 0.5),
    (MemoryType.PROJECT, "评审会议", "下周三有代码评审会", "下周三下午两点有代码评审会议，需要讲解记忆系统的存储层设计", 0.5),
    (MemoryType.PROJECT, "招实习生", "团队要招一个AI实习生", "用户所在团队要招一个AI方向的实习生，用户负责面试技术部分", 0.3),
    (MemoryType.PROJECT, "整理Prompt模板", "整理记忆提取的Prompt", "在整理记忆提取环节使用的 Prompt 模板，尝试提升提取准确率", 0.5),
    (MemoryType.PROJECT, "BM25权重调优", "在调混合检索权重", "在调整 BM25 和向量检索的混合权重，目前是 0.3:0.7，考虑改为 0.4:0.6", 0.4),
    (MemoryType.PROJECT, "测试覆盖率", "补测试覆盖率", "记忆系统目前测试覆盖率约 60%，目标提升到 80% 以上", 0.4),
    (MemoryType.PROJECT, "安全审计", "在做安全合规审查", "公司安全部门要求对记忆系统做数据安全审计，需要提供数据加密和脱敏方案", 0.5),
    (MemoryType.PROJECT, "给咪咪买猫爬架", "给猫买了猫爬架", "给咪咪在京东买了一个三层猫爬架，咪咪很喜欢趴在最高层", 0.2),

    # 更多个人细节
    (MemoryType.USER, "穿衣风格休闲", "用户喜欢休闲穿搭", "用户平时穿衣偏休闲风，喜欢穿卫衣和牛仔裤，不喜欢正装", 0.3),
    (MemoryType.USER, "B站粉丝", "用户常看B站", "用户经常看B站，关注了很多技术UP主，最喜欢的是「稚晖君」", 0.3),
    (MemoryType.USER, "支付宝刷地铁", "用户用支付宝坐地铁", "用户每天坐地铁上班，用支付宝刷码进站，从体育西到珠江新城", 0.3),
    (MemoryType.USER, "腰椎不好", "用户腰椎有点问题", "用户因为久坐腰椎不太好，买了人体工学椅，偶尔还是会酸痛", 0.4),
    (MemoryType.USER, "用VSCode", "用户用VSCode开发", "用户日常用 VS Code 开发，装了 Python、Copilot、GitLens 插件", 0.4),
    (MemoryType.USER, "晚睡晚起", "用户习惯晚睡晚起", "用户习惯凌晨一点睡觉，早上九点半到公司，公司弹性工作制", 0.3),
    (MemoryType.USER, "戴Apple Watch", "用户戴Apple Watch", "用户戴 Apple Watch Ultra 2，主要用来监测运动和睡眠数据", 0.3),
    (MemoryType.USER, "喜欢拍照", "用户喜欢手机摄影", "用户喜欢拍照，尤其喜欢拍城市夜景和猫咪，照片发在朋友圈和小红书", 0.3),
    (MemoryType.USER, "看NBA", "用户喜欢看NBA比赛", "用户喜欢看 NBA 比赛，支持勇士队，每周都会看直播或回放", 0.4),

    # 更多行为指导
    (MemoryType.FEEDBACK, "错误处理要完善", "try-except要有具体异常", "写 try-except 时要捕获具体异常类型，不要用裸 except\n原因：裸except会吞掉所有错误\n应用场景：异常处理代码", 0.6),
    (MemoryType.FEEDBACK, "类型提示要加", "函数要加type hints", "函数参数和返回值要加类型提示 type hints\n原因：方便IDE推断和代码审查\n应用场景：所有Python函数", 0.5),
    (MemoryType.FEEDBACK, "不要打印密码", "日志不要打印敏感信息", "日志中不要打印密码、token、API key 等敏感信息\n原因：安全合规要求\n应用场景：所有日志输出", 0.7),
    (MemoryType.FEEDBACK, "文件不超500行", "单文件不超过500行", "单个 Python 文件代码不要超过 500 行，超了就拆分\n原因：方便代码审查\n应用场景：文件组织", 0.4),
    (MemoryType.FEEDBACK, "PR要关联issue", "PR要关联对应issue", "提交 PR 时要在描述中关联对应的 Linear issue 编号\n原因：方便追踪\n应用场景：所有PR", 0.4),

    # 更多外部资源
    (MemoryType.REFERENCE, "Docker私有仓库", "Docker镜像仓库", "公司 Docker 私有仓库 https://harbor.internal.com，推送需要先登录", 0.3),
    (MemoryType.REFERENCE, "Rust官方教程", "Rust学习资源", "Rust 官方教程 https://doc.rust-lang.org/book/，用户正在看", 0.3),
    (MemoryType.REFERENCE, "掘金账号", "用户的掘金主页", "用户掘金主页 https://juejin.cn/user/xiaozhao，写技术博客的地方", 0.3),
    (MemoryType.REFERENCE, "多邻国学日语", "日语学习APP", "用户在多邻国APP上学日语，每天打卡，已经连续30天", 0.3),
    (MemoryType.REFERENCE, "知乎专栏", "用户知乎专栏", "用户在知乎有个专栏「AI产品笔记」，写AI产品相关文章", 0.3),
    (MemoryType.REFERENCE, "装修参考小红书", "装修灵感在小红书", "装修参考灵感主要从小红书收藏夹里找，收藏了200多条极简风格案例", 0.2),

    # 对话原文补充
    (MemoryType.PROJECT, "讨论线程安全", "讨论了ThreadPoolExecutor方案", "用户和AI讨论了后台记忆提取的线程安全问题，决定用 ThreadPoolExecutor(max_workers=1) 替代裸线程", 0.4),
    (MemoryType.PROJECT, "讨论Token预算", "讨论了prompt长度控制", "讨论了 system prompt 的 token 预算管理，设置了 core/recalled/pack/history 四段预算上限", 0.4),
    (MemoryType.PROJECT, "讨论类型分类", "讨论了记忆四分类方案", "讨论了记忆类型分类方案，参考 Claude Code 设计了 user/feedback/project/reference 四种类型", 0.4),
    (MemoryType.PROJECT, "火锅店推荐", "推荐了几家火锅店", "用户问了广州好吃的火锅店，AI推荐了海底捞天河店和谭鸭血珠江新城店", 0.2),
    (MemoryType.PROJECT, "帮写周报", "帮用户写了周报", "帮用户写了一份周报，内容包括记忆系统优化改造进展和下周计划", 0.2),
    (MemoryType.PROJECT, "讨论微服务拆分", "讨论了是否拆微服务", "用户问是否需要把记忆系统拆成微服务架构，AI建议先保持单体，后续视规模再拆", 0.3),
    (MemoryType.USER, "吐槽加班", "用户最近加班多", "用户吐槽最近加班很多，连续两周每天都到晚上九点才下班", 0.3),
    (MemoryType.USER, "想去潜水", "想学潜水", "用户说等去日本旅游时想在冲绳体验潜水，之前没试过", 0.2),
    (MemoryType.USER, "喜欢逛宜家", "周末常逛宜家", "用户和小美周末经常逛宜家，买一些家居用品，装修后需要的东西很多", 0.2),
    (MemoryType.USER, "鞋子穿43码", "用户鞋码43", "用户鞋子穿 43 码，喜欢穿 Nike 和 New Balance 的运动鞋", 0.2),

    # ═══════════════════════════════════════════
    # 最后一批补充到 200 条
    # ═══════════════════════════════════════════
    (MemoryType.USER, "血型O型", "用户O型血", "用户是O型血，献过两次血", 0.2),
    (MemoryType.USER, "用Mac办公", "工作电脑MacBook", "用户工作电脑是 MacBook Pro 14寸 M3 Pro，正在等 M4 到货后换掉", 0.3),
    (MemoryType.USER, "Spotify听歌", "用Spotify听音乐", "用户用 Spotify 听歌，有 Premium 会员，日常听华语流行和日系City Pop", 0.3),
    (MemoryType.USER, "左撇子", "用户是左撇子", "用户是左撇子，写字和吃饭都用左手，小时候被纠正过但没改过来", 0.2),
    (MemoryType.USER, "怕蛇", "用户怕蛇", "用户非常怕蛇，看到蛇的图片都会紧张", 0.2),
    (MemoryType.PROJECT, "准备PMP考试", "在备考PMP", "用户在备考 PMP 项目管理证书，买了教材和网课，计划年底考", 0.4),
    (MemoryType.PROJECT, "研究LangChain", "在看LangChain源码", "用户在研究 LangChain 的 Memory 模块实现，想借鉴一些设计思路", 0.5),
    (MemoryType.PROJECT, "给父亲买按摩椅", "要给爸买按摩椅", "用户打算父亲节给爸买个按摩椅，在京东看了荣泰和奥佳华", 0.3),
    (MemoryType.PROJECT, "小美学画画", "小美在学水彩画", "女友小美最近报了个水彩画班，每周末去上课，用户有时候陪她", 0.3),
    (MemoryType.PROJECT, "换窗帘", "新房要买窗帘", "新房需要定制窗帘，在宜家和网上对比，预算三千左右", 0.2),
    (MemoryType.FEEDBACK, "不要推荐付费工具", "不推荐付费工具", "不要推荐收费的商业软件工具，优先推荐开源免费的\n原因：公司预算审批流程很长\n应用场景：工具推荐", 0.5),
    (MemoryType.FEEDBACK, "修改前先备份", "代码改之前先备份", "修改重要代码前先 git stash 或创建分支备份\n原因：有过覆盖错误经历\n应用场景：代码修改", 0.5),
    (MemoryType.FEEDBACK, "不要用print调试", "不用print调试", "调试不要用 print，用 logging 或 debugger\n原因：print 容易遗留在生产代码中\n应用场景：调试代码", 0.5),
    (MemoryType.REFERENCE, "PMP教材", "PMP备考资料", "PMP 备考用的是《PMBOK指南第七版》和光环国际的网课", 0.3),
    (MemoryType.REFERENCE, "LangChain文档", "LangChain官方文档", "LangChain 文档在 https://python.langchain.com/docs/", 0.3),
    (MemoryType.REFERENCE, "搬家公司", "搬家公司联系方式", "搬家公司用的是「货拉拉」APP，也在看「蚂蚁搬家」", 0.2),
    (MemoryType.REFERENCE, "牙科诊所", "隐适美牙科诊所", "做隐适美的诊所是「瑞尔齿科」天河店，主治医生姓林", 0.3),
    (MemoryType.REFERENCE, "驾校电话", "驾校联系方式", "驾校是广州猎豹驾校黄埔分校，教练姓周，手机 159xxxxxxxx", 0.2),
    (MemoryType.PROJECT, "年度总结PPT", "要写年度工作总结", "年底要写年度工作总结PPT，需要列出今年做的核心项目和成果", 0.4),
    (MemoryType.PROJECT, "做体检", "公司组织年度体检", "公司下个月组织年度体检，地点在爱康国宾天河店", 0.3),
    (MemoryType.PROJECT, "清理照片", "手机存储快满了", "用户手机 256GB 存储快满了，需要清理照片和视频到NAS上", 0.2),
    (MemoryType.USER, "喜欢喝柠檬茶", "爱喝柠檬茶", "用户夏天喜欢喝手打柠檬茶，最喜欢的店是「丘大叔柠檬茶」", 0.3),
    (MemoryType.USER, "看《三体》三遍", "最喜欢的小说三体", "用户最喜欢的小说是刘慈欣的《三体》三部曲，看了三遍", 0.4),
    (MemoryType.USER, "乳糖不耐受", "不能喝纯牛奶", "用户有轻微乳糖不耐受，喝纯牛奶会拉肚子，只能喝酸奶", 0.4),
    (MemoryType.USER, "戴隐形眼镜", "偶尔戴隐形眼镜", "用户运动和约会时会戴日抛隐形眼镜，平时戴框架眼镜", 0.3),
    (MemoryType.PROJECT, "帮同事搬家", "帮同事搬了家", "上周六帮同事老王搬家，从天河搬到番禺，累了一整天", 0.2),
    (MemoryType.PROJECT, "买了Switch", "入手了Switch", "用户上个月买了任天堂 Switch OLED版，在玩塞尔达和动物森友会", 0.3),
    (MemoryType.PROJECT, "给咪咪剪指甲", "给猫剪指甲", "用户学会了给咪咪剪指甲，每两周剪一次，咪咪很不配合", 0.2),
    (MemoryType.USER, "讨厌下雪", "不喜欢下雪天", "用户不喜欢下雪天，觉得冷而且路滑，庆幸广州几乎不下雪", 0.2),
    (MemoryType.USER, "喜欢逛书店", "周末逛书店", "用户偶尔周末去方所书店或1200bookshop逛，觉得很放松", 0.3),
    (MemoryType.USER, "睡眠追踪", "用Apple Watch追踪睡眠", "用户用 Apple Watch 追踪睡眠质量，平均每晚深睡只有1.5小时", 0.3),
]

# 干扰记忆：内容相似但不相关的记忆，测试系统是否会误召回
NOISE_MEMORIES: list[tuple[MemoryType, str, str, str, float]] = [
    (MemoryType.PROJECT, "其他用户的猫", "另一个用户有只猫", "张三家里养了一只波斯猫，喜欢晒太阳", 0.3),
    (MemoryType.USER, "其他用户爱美食", "另一个用户喜欢烤肉", "李四最喜欢吃韩式烤肉，每周都要去一次", 0.3),
    (MemoryType.PROJECT, "其他项目", "另一个推荐系统项目", "王五在做一个电商推荐系统，基于协同过滤算法", 0.3),
    (MemoryType.FEEDBACK, "其他规则", "另一个用户的偏好", "请用英文回答所有问题，因为我在练习英文", 0.3),
    (MemoryType.USER, "其他城市", "另一个用户在北京", "赵六住在北京朝阳区，在一家互联网大厂工作", 0.3),
]

# 查询 → 期望命中的记忆 name 列表
# (query, expected_hits: 期望在 top-5 中出现的记忆 name, must_not_hit: 不应出现的 name)
RETRIEVAL_TESTS: list[tuple[str, list[str], list[str]]] = [
    # ═══════════════════════════════════════════
    # 用户身份类查询
    # ═══════════════════════════════════════════
    ("你叫什么名字？", ["姓名赵云"], []),
    ("你住在哪里？", ["广州居住"], ["其他城市"]),
    ("你是做什么工作的？", ["产品开发"], []),
    ("你用什么编程语言？", ["Python技术栈"], []),
    ("你生日是几号？", ["生日九月"], []),
    ("你老家在哪？", ["老家湖南"], []),
    ("你多高？", ["身高175"], []),
    ("你哪个大学毕业的？", ["本科华工"], []),
    ("你开什么车？", ["开本田思域"], []),
    ("你用什么手机？", ["用iPhone"], []),

    # ═══════════════════════════════════════════
    # 个人喜好与习惯
    # ═══════════════════════════════════════════
    ("你喜欢吃什么？", ["喜欢火锅"], ["其他用户爱美食"]),
    ("推荐一首歌给我", ["周杰伦粉丝"], []),
    ("你有宠物吗？", ["养猫"], ["其他用户的猫"]),
    ("你女朋友是谁？", ["女友小美"], []),
    ("你有什么运动爱好？", ["健身爱好"], []),
    ("你喝酒吗？", ["不喝酒"], []),
    ("你喝咖啡吗？", ["喝美式咖啡"], []),
    ("你有什么食物忌口？", ["不吃香菜"], []),
    ("你喜欢看什么电影？", ["喜欢看科幻电影"], []),
    ("你平时玩什么游戏？", ["玩原神"], []),
    ("你对什么东西过敏？", ["过敏体质"], []),
    ("春天花粉过敏怎么办？", ["有花粉过敏"], []),

    # ═══════════════════════════════════════════
    # 家庭关系
    # ═══════════════════════════════════════════
    ("你弟弟在做什么？", ["弟弟在读大学"], []),
    ("你爸妈怎么样？", ["父亲退休"], []),
    ("你妈做什么菜好吃？", ["母亲做菜好"], []),
    ("你弟弟考研考哪个学校？", ["弟弟考研"], []),

    # ═══════════════════════════════════════════
    # 项目和工作
    # ═══════════════════════════════════════════
    ("记忆系统开发进度怎样？", ["记忆系统开发"], ["其他项目"]),
    ("OpenClaw是什么？", ["OpenClaw研究"], []),
    ("周五演示准备得怎样？", ["周五演示"], []),
    ("部署方案是什么？", ["部署方案"], []),
    ("ONNX优化效果如何？", ["性能优化"], []),
    ("下周技术分享准备好了吗？", ["团队分享准备"], []),
    ("API接口是怎么设计的？", ["API接口设计"], []),
    ("有没有做压力测试？", ["压力测试计划"], []),
    ("数据迁移脚本写了吗？", ["数据迁移脚本"], []),
    ("Rust学得怎么样？", ["学Rust"], []),
    ("RAG技术有研究进展吗？", ["研究RAG"], []),
    ("最近工作有什么变化？", ["考虑跳槽"], []),
    ("你打算做自媒体？", ["做副业"], []),
    ("新电脑到了吗？", ["MacBook到货"], []),
    ("Claude Code优化改造做了哪些？", ["记忆系统优化改造"], []),

    # ═══════════════════════════════════════════
    # 行为指导
    # ═══════════════════════════════════════════
    ("请说英文", ["用中文回答"], ["其他规则"]),
    ("帮我写个单元测试", ["不要mock数据库", "推荐用pytest"], []),
    ("这个功能怎么实现？", ["先给方案再写代码"], []),
    ("帮我实现一个缓存方案", ["不推荐Redis"], []),
    ("帮我定义一个数据结构", ["用dataclass"], []),
    ("写个异步后台任务", ["异步用threading"], []),
    ("帮我写SQL查询", ["SQL不用ORM"], []),
    ("帮我提交下代码", ["Git提交要规范"], []),
    ("变量名怎么命名？", ["变量名用英文"], []),
    ("这段代码能不能重构一下？", ["不要过度抽象", "不要删除代码"], []),
    ("帮我加个日志", ["日志用中文"], []),
    ("配置参数怎么管理？", ["配置用环境变量"], []),

    # ═══════════════════════════════════════════
    # 外部资源
    # ═══════════════════════════════════════════
    ("API文档在哪？", ["API文档"], []),
    ("哪里看bug列表？", ["Linear看板"], []),
    ("监控看板地址", ["Grafana监控"], []),
    ("需求文档在哪看？", ["飞书文档"], []),
    ("代码仓库在哪？", ["GitHub仓库"], []),
    ("线上报错去哪看？", ["Sentry错误追踪"], []),
    ("CI/CD怎么配置的？", ["Jenkins构建"], []),
    ("架构设计文档在哪？", ["Confluence文档"], []),
    ("embedding模型文档在哪？", ["sentence-transformers文档"], []),
    ("FastAPI文档网址", ["FastAPI文档"], []),
    ("公司VPN怎么连？", ["公司VPN"], []),

    # ═══════════════════════════════════════════
    # 日常生活
    # ═══════════════════════════════════════════
    ("驾照考试怎么样了？", ["驾照科目三"], []),
    ("最近在看什么书？", ["读书计划"], []),
    ("最近睡眠怎么样？", ["失眠问题"], []),
    ("新房装修进度", ["装修房子"], []),
    ("国庆有什么计划？", ["年假计划"], []),
    ("日本签证办了吗？", ["签证办理"], []),
    ("小美生日送什么好？", ["小美生日礼物"], []),
    ("搬家安排怎么样？", ["搬家计划"], []),
    ("猫咪疫苗打了吗？", ["咪咪疫苗"], []),
    ("投影仪选好了吗？", ["考虑买投影仪"], []),
    ("牙齿矫正还要多久？", ["牙齿矫正"], []),
    ("NAS组装得怎样？", ["组装NAS"], []),
    ("同学聚会去不去？", ["高中同学聚会"], []),
    ("腰伤好了吗？", ["健身受伤"], []),
    ("日语学到什么水平了？", ["日语学习"], []),
    ("周末有什么安排？", ["约朋友烧烤"], []),
    ("减脂效果怎么样？", ["减脂计划"], []),

    # ═══════════════════════════════════════════
    # 复合/模糊/间接查询（高难度）
    # ═══════════════════════════════════════════
    ("周末去珠海怎么安排？", ["珠海长隆"], []),
    ("帮我写段Python代码并测试", ["Python技术栈", "代码要注释"], []),
    ("你们公司用什么技术栈？", ["Python技术栈"], []),
    ("你打算什么时候求婚？", ["准备求婚"], []),
    ("帮弟弟复习什么科目？", ["弟弟考研"], []),
    ("年会准备什么节目？", ["公司年会节目"], []),
    ("咪咪最近怎么样？", ["养猫"], []),
    ("你眼睛怎么样？", ["近视400度"], []),
    ("健身房在哪？", ["健身房地址"], []),
    ("宠物医院电话多少？", ["宠物医院电话"], []),
]


# ══════════════════════════════════════════════════════════
# 测试主体
# ══════════════════════════════════════════════════════════

def _setup_store_and_searcher():
    """初始化 store、embedder、searcher，并灌入全部测试数据"""
    store = SQLiteMemoryStore(db_path=":memory:")
    store.init()
    embedder = LocalEmbeddingProvider()

    user_id = "test-user"
    noise_user_id = "other-user"  # 干扰数据归属不同用户，验证 user_id 隔离

    total = len(SEED_MEMORIES) + len(NOISE_MEMORIES)
    print(f"\n灌入 {total} 条记忆（{len(SEED_MEMORIES)} 正常 + {len(NOISE_MEMORIES)} 干扰/其他用户）...")
    t0 = time.time()
    for mtype, name, desc, content, imp in SEED_MEMORIES:
        vec = embedder.embed(content)
        rec = MemoryRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            embedding=vec,
            memory_type=mtype,
            name=name,
            description=desc,
            importance=imp,
        )
        store.insert_memory(rec)
        store.fts_sync(rec.id, content)
    # 干扰数据存到其他用户名下 — 检索时应完全不可见
    for mtype, name, desc, content, imp in NOISE_MEMORIES:
        vec = embedder.embed(content)
        rec = MemoryRecord(
            id=str(uuid.uuid4()),
            user_id=noise_user_id,
            content=content,
            embedding=vec,
            memory_type=mtype,
            name=name,
            description=desc,
            importance=imp,
        )
        store.insert_memory(rec)
        store.fts_sync(rec.id, content)
    elapsed = time.time() - t0
    print(f"灌入完成，耗时 {elapsed:.2f}s（{total/elapsed:.0f} 条/秒）")

    # 启用 Cross-Encoder reranker 精排
    from memory_agent.providers.reranker_local import LocalRerankerProvider
    reranker = LocalRerankerProvider()

    searcher = MemorySearcher(store, embedder, reranker=reranker)
    return store, searcher, user_id


def test_retrieval_accuracy():
    """检索召回准确率测试"""
    store, searcher, user_id = _setup_store_and_searcher()

    total_queries = len(RETRIEVAL_TESTS)
    hit_count = 0           # 期望记忆出现在 top-5 的次数
    total_expected = 0      # 期望命中的总条数
    false_positive = 0      # 不应出现但出现的次数
    detailed_results = []

    print(f"\n{'='*70}")
    print(f"开始检索测试：{total_queries} 条查询")
    print(f"{'='*70}\n")

    for query, expected_names, must_not_names in RETRIEVAL_TESTS:
        recalled, packs = searcher.search(user_id, query)
        recalled_names = {_find_name(r.content) for r in recalled}
        recalled_contents = [r.content[:60] for r in recalled]

        # 检查期望命中
        query_hits = 0
        for exp_name in expected_names:
            total_expected += 1
            # 在召回结果中查找包含该 name 对应内容的记忆
            found = _check_hit(exp_name, recalled)
            if found:
                hit_count += 1
                query_hits += 1

        # 检查不应命中
        query_fp = 0
        for bad_name in must_not_names:
            if _check_hit(bad_name, recalled):
                false_positive += 1
                query_fp += 1

        # 状态标记
        all_hit = query_hits == len(expected_names)
        no_fp = query_fp == 0
        status = "PASS" if (all_hit and no_fp) else "FAIL"

        detailed_results.append({
            "query": query,
            "status": status,
            "expected": expected_names,
            "hits": query_hits,
            "fp": query_fp,
            "recalled": recalled_contents,
        })

        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} [{status}] \"{query}\"")
        if not all_hit:
            missed = [n for n in expected_names if not _check_hit(n, recalled)]
            print(f"      未命中: {missed}")
        if query_fp > 0:
            bad_hits = [n for n in must_not_names if _check_hit(n, recalled)]
            print(f"      误召回: {bad_hits}")
        if not all_hit:
            print(f"      实际召回: {recalled_contents[:3]}")

    # ── 汇总报告 ──
    recall_rate = hit_count / total_expected * 100 if total_expected > 0 else 0
    pass_count = sum(1 for r in detailed_results if r["status"] == "PASS")

    print(f"\n{'='*70}")
    print(f"检索召回测试报告")
    print(f"{'='*70}")
    print(f"  查询总数:     {total_queries}")
    print(f"  通过查询:     {pass_count}/{total_queries} ({pass_count/total_queries*100:.1f}%)")
    print(f"  召回命中:     {hit_count}/{total_expected} ({recall_rate:.1f}%)")
    print(f"  误召回:       {false_positive}")
    print(f"{'='*70}")

    # 断言：召回率应 >= 70%（混合检索不带 reranker 的基线）
    assert recall_rate >= 70, f"召回率 {recall_rate:.1f}% 低于 70% 基线"
    # 断言：误召回应 <= 3
    assert false_positive <= 3, f"误召回 {false_positive} 次，超过 3 次上限"

    print(f"\n  结论: 召回率 {recall_rate:.1f}% >= 70% 基线，误召回 {false_positive} <= 3 ✓\n")


# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════

# name → content 映射（从 SEED + NOISE 构建）
_NAME_TO_CONTENT: dict[str, str] = {}
for _all in [SEED_MEMORIES, NOISE_MEMORIES]:
    for _, _name, _, _content, _ in _all:
        _NAME_TO_CONTENT[_name] = _content


def _check_hit(name: str, recalled: list) -> bool:
    """检查 name 对应的记忆是否在召回结果中"""
    target_content = _NAME_TO_CONTENT.get(name, "")
    if not target_content:
        return False
    for r in recalled:
        # 内容前 40 字符匹配即认为命中
        if target_content[:40] in r.content or r.content[:40] in target_content:
            return True
    return False


def _find_name(content: str) -> str:
    """从 content 反查 name"""
    for name, c in _NAME_TO_CONTENT.items():
        if c[:40] in content or content[:40] in c:
            return name
    return content[:30]


if __name__ == "__main__":
    test_retrieval_accuracy()
