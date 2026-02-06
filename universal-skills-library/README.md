# 🚀 Universal Skills Library

一个精心整理的高质量Python工具库Skills集合，包含25+个实用技能模块，涵盖数据处理、机器学习、Web开发、DevOps等多个领域。

## 📖 项目简介

本项目收集并整理了GitHub上最受欢迎的Python工具库，为每个工具创建了详细的使用文档（Skills），包括:
- 工具简介和使用场景
- 安装和配置指南
- 完整的代码示例
- 最佳实践和设计模式
- 常见问题解决方案

每个skill都经过精心编写,帮助开发者快速上手并掌握各类工具的核心功能。

## 🎯 项目目标

- ✅ 提供一站式的工具学习资源
- ✅ 帮助开发者快速选择合适的工具
- ✅ 分享最佳实践和避坑指南
- ✅ 建立完整的技术栈知识体系

## 📊 Skills分类

### 数据处理类 (Data Processing)
| Skill | GitHub Stars | 用途 | 文档 |
|-------|-------------|------|------|
| pandas-skill | 25K+ | 数据分析和处理 | [查看](pandas-skill/SKILL.md) |
| polars-skill | 快速增长 | 高性能数据处理 | [查看](polars-skill/SKILL.md) |
| numpy-skill | 核心库 | 数值计算 | [查看](numpy-skill/SKILL.md) |
| dask-skill | - | 分布式计算 | [查看](dask-skill/SKILL.md) |

### 机器学习/AI类 (Machine Learning & AI)
| Skill | GitHub Stars | 用途 | 文档 |
|-------|-------------|------|------|
| pytorch-skill | 领先 | 深度学习框架 | [查看](pytorch-skill/SKILL.md) |
| tensorflow-skill | Google | 生产级深度学习 | [查看](tensorflow-skill/SKILL.md) |
| scikit-learn-skill | 传统ML | 机器学习算法 | [查看](scikit-learn-skill/SKILL.md) |
| huggingface-skill | 120K+ | Transformer模型 | [查看](huggingface-skill/SKILL.md) |

### 数据可视化类 (Data Visualization)
| Skill | GitHub Stars | 用途 | 文档 |
|-------|-------------|------|------|
| matplotlib-skill | 最广泛 | 静态图表 | [查看](matplotlib-skill/SKILL.md) |
| seaborn-skill | 高级 | 统计可视化 | [查看](seaborn-skill/SKILL.md) |
| plotly-skill | 交互式 | Web图表 | [查看](plotly-skill/SKILL.md) |

### Web开发类 (Web Development)
| Skill | GitHub Stars | 用途 | 文档 |
|-------|-------------|------|------|
| fastapi-skill | 94K+ | 现代API框架 | [查看](fastapi-skill/SKILL.md) |
| django-skill | 76K+ | 全栈框架 | [查看](django-skill/SKILL.md) |
| flask-skill | 66K+ | 轻量级框架 | [查看](flask-skill/SKILL.md) |
| beautifulsoup-skill | - | 网页解析 | [查看](beautifulsoup-skill/SKILL.md) |
| requests-skill | - | HTTP请求 | [查看](requests-skill/SKILL.md) |

### DevOps/自动化类 (DevOps & Automation)
| Skill | GitHub Stars | 用途 | 文档 |
|-------|-------------|------|------|
| docker-skill | 标准 | 容器化 | [查看](docker-skill/SKILL.md) |
| kubernetes-skill | 标准 | 容器编排 | [查看](kubernetes-skill/SKILL.md) |
| ansible-skill | - | 配置管理 | [查看](ansible-skill/SKILL.md) |
| terraform-skill | IaC | 基础设施即代码 | [查看](terraform-skill/SKILL.md) |

### 自然语言处理类 (NLP)
| Skill | GitHub Stars | 用途 | 文档 |
|-------|-------------|------|------|
| spacy-skill | 生产级 | NLP处理 | [查看](spacy-skill/SKILL.md) |
| nltk-skill | 教学 | NLP基础 | [查看](nltk-skill/SKILL.md) |
| transformers-skill | 120K+ | 预训练模型 | [查看](transformers-skill/SKILL.md) |

### 其他工具类 (Utilities)
| Skill | GitHub Stars | 用途 | 文档 |
|-------|-------------|------|------|
| pytest-skill | - | 单元测试 | [查看](pytest-skill/SKILL.md) |
| jupyter-skill | - | 交互式开发 | [查看](jupyter-skill/SKILL.md) |

## 🚀 快速开始

### 1. 浏览Skills目录

查看 [SKILLS_CATALOG.md](SKILLS_CATALOG.md) 获取所有skills的完整列表和分类。

### 2. 选择你需要的Skill

根据你的需求,在上表中找到对应的skill,点击"查看"链接。

### 3. 学习和实践

每个skill文档包含:
- 📚 **工具简介**: 了解工具的核心特性和适用场景
- 🔧 **安装配置**: 详细的安装步骤和环境配置
- 💻 **代码示例**: 从基础到高级的完整示例
- 🎯 **最佳实践**: 行业推荐的使用方法
- ⚠️ **常见问题**: 避坑指南和问题解决

### 4. 实际应用

将学到的知识应用到你的项目中,遇到问题可以参考文档的"常见问题"部分。

## 📂 项目结构

```
universal-skills-library/
├── README.md                    # 项目主文档
├── SKILLS_CATALOG.md           # Skills完整目录
├── SKILL_TEMPLATE.md           # Skill文档模板
│
├── pandas-skill/               # Pandas技能模块
│   ├── SKILL.md               # 详细文档
│   ├── examples/              # 代码示例
│   └── best-practices.md      # 最佳实践
│
├── pytorch-skill/             # PyTorch技能模块
│   ├── SKILL.md
│   ├── examples/
│   └── tutorials/
│
├── fastapi-skill/             # FastAPI技能模块
│   ├── SKILL.md
│   └── examples/
│
└── [其他skills]/
```

## 🛠️ 使用场景示例

### 场景1: 数据分析项目

```
1. 使用 pandas-skill 进行数据清洗和处理
2. 使用 matplotlib-skill 或 plotly-skill 进行可视化
3. 使用 jupyter-skill 创建交互式分析报告
```

### 场景2: 机器学习项目

```
1. 使用 pandas-skill 准备数据
2. 使用 scikit-learn-skill 训练传统ML模型
3. 使用 pytorch-skill 或 tensorflow-skill 训练深度学习模型
4. 使用 fastapi-skill 部署模型API
```

### 场景3: Web应用开发

```
1. 使用 fastapi-skill 或 django-skill 构建后端API
2. 使用 requests-skill 处理HTTP请求
3. 使用 beautifulsoup-skill 进行数据爬取
4. 使用 docker-skill 容器化应用
5. 使用 kubernetes-skill 部署到生产环境
```

## 📈 统计信息

- **总Skills数量**: 25个
- **涵盖类别**: 7大类
- **代码示例**: 200+个
- **最佳实践**: 100+条
- **常见问题**: 50+个

## 🤝 贡献指南

欢迎贡献新的skills或改进现有文档!

### 如何贡献

1. Fork本项目
2. 使用 `SKILL_TEMPLATE.md` 创建新的skill文档
3. 确保包含:
   - 详细的工具简介
   - 完整的安装步骤
   - 至少5个代码示例
   - 最佳实践建议
   - 常见问题解答
4. 提交Pull Request

### 文档规范

- 使用中文编写
- 代码示例需要可运行
- 提供实际应用场景
- 注明工具版本和更新日期

## 📜 许可证

本项目采用 MIT 许可证。

## 🔗 相关资源

### 数据来源
本项目的工具选择基于以下来源的研究:
- [NumPy vs Pandas vs Polars](https://www.index.dev/skill-vs-skill/ai-numpy-vs-pandas-vs-polars)
- [Top Python Libraries for Data Science](https://www.datacamp.com/blog/top-python-libraries-for-data-science)
- [Best Web Frameworks](https://github.com/ml-tooling/best-of-web-python)
- [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
- [Top DevOps Tools](https://titanapps.io/blog/10-best-devops-automation-tools/)
- [NLP Libraries Guide](https://milvus.io/ai-quick-reference/what-are-the-most-popular-nlp-libraries)

### 学习资源
- [Python官方文档](https://docs.python.org/)
- [Real Python](https://realpython.com/)
- [Awesome Python](https://github.com/vinta/awesome-python)

## 📧 联系方式

如有问题或建议,欢迎:
- 提交Issue
- 发起Discussion
- 贡献Pull Request

## ⭐ Star History

如果这个项目对你有帮助,请给个Star⭐支持一下!

---

**最后更新**: 2026-01-22
**版本**: 1.0.0
**维护者**: Universal Skills Team
