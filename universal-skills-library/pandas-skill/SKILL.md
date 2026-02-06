# Pandas Skill

## 📚 工具简介

**Pandas** 是Python中最流行的数据分析和处理库，提供了高性能、易用的数据结构和数据分析工具。

### 核心特性
- **DataFrame和Series**: 强大的数据结构
- **数据清洗**: 处理缺失值、重复数据
- **数据转换**: 灵活的数据操作和变换
- **数据聚合**: 分组、透视表等高级功能
- **时间序列**: 专业的时间序列处理能力
- **IO工具**: 支持多种格式(CSV, Excel, SQL, JSON等)

### GitHub信息
- **Stars**: 25,000+
- **下载量**: 24亿+
- **仓库**: https://github.com/pandas-dev/pandas
- **官方文档**: https://pandas.pydata.org/

### 适用场景
✅ 中小型数据集分析 (< 1GB)
✅ 数据清洗和预处理
✅ 探索性数据分析(EDA)
✅ 财务分析和报表
✅ 时间序列分析

❌ 超大数据集 (考虑Polars或Dask)
❌ 需要极致性能的生产环境

---

## 🔧 安装和配置

### 基础安装

```bash
# 使用pip安装
pip install pandas --break-system-packages

# 安装完整版本(包含所有可选依赖)
pip install pandas[all] --break-system-packages

# 安装特定版本
pip install pandas==2.2.0 --break-system-packages
```

### 常用依赖

```bash
# Excel支持
pip install openpyxl xlrd --break-system-packages

# 数据库支持
pip install sqlalchemy psycopg2-binary --break-system-packages

# 高性能计算
pip install numpy numexpr bottleneck --break-system-packages

# 可视化
pip install matplotlib seaborn --break-system-packages
```

### 验证安装

```python
import pandas as pd
print(f"Pandas version: {pd.__version__}")

# 查看配置
pd.show_versions()
```

---

## 💻 代码示例

### 1. 基础数据操作

```python
import pandas as pd
import numpy as np

# 创建DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000],
    'department': ['HR', 'IT', 'IT', 'Sales']
})

# 查看数据
print(df.head())
print(df.info())
print(df.describe())

# 选择数据
print(df['name'])  # 选择列
print(df[df['age'] > 28])  # 条件筛选
print(df.loc[0:2, ['name', 'age']])  # 标签索引
print(df.iloc[0:2, 0:2])  # 位置索引
```

### 2. 数据清洗

```python
# 创建包含缺失值的数据
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

# 处理缺失值
df_dropna = df.dropna()  # 删除含缺失值的行
df_fillna = df.fillna(0)  # 填充缺失值
df_fillna_mean = df.fillna(df.mean())  # 用均值填充

# 检测缺失值
print(df.isnull().sum())

# 删除重复值
df_unique = df.drop_duplicates()

# 数据类型转换
df['A'] = df['A'].astype(int)
```

### 3. 数据转换和聚合

```python
# 分组聚合
grouped = df.groupby('department').agg({
    'salary': ['mean', 'sum', 'count'],
    'age': 'mean'
})

# 透视表
pivot = df.pivot_table(
    values='salary',
    index='department',
    aggfunc=['mean', 'sum']
)

# 数据合并
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# 内连接
inner = pd.merge(df1, df2, on='key', how='inner')
# 外连接
outer = pd.merge(df1, df2, on='key', how='outer')

# 拼接
concatenated = pd.concat([df1, df2], axis=0)
```

### 4. 时间序列处理

```python
# 创建时间序列
dates = pd.date_range('2024-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# 时间索引
print(ts['2024-01'])  # 选择特定月份
print(ts['2024-01-01':'2024-01-10'])  # 时间范围

# 重采样
monthly = ts.resample('M').mean()  # 按月聚合
weekly = ts.resample('W').sum()  # 按周求和

# 时间窗口
rolling_mean = ts.rolling(window=7).mean()  # 7天移动平均
```

### 5. 读写文件

```python
# CSV
df.to_csv('data.csv', index=False)
df = pd.read_csv('data.csv')

# Excel
df.to_excel('data.xlsx', sheet_name='Sheet1', index=False)
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df.to_json('data.json', orient='records')
df = pd.read_json('data.json')

# SQL
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.db')
df.to_sql('table_name', engine, if_exists='replace')
df = pd.read_sql('SELECT * FROM table_name', engine)

# Parquet (高效的列式存储)
df.to_parquet('data.parquet')
df = pd.read_parquet('data.parquet')
```

---

## 🎯 最佳实践

### 1. 性能优化

```python
# 使用向量化操作,避免循环
# ❌ 不好的做法
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'A'] * 2

# ✅ 好的做法
df['new_col'] = df['A'] * 2

# 使用类别类型节省内存
df['category_col'] = df['category_col'].astype('category')

# 分块读取大文件
chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)

# 使用query方法提高可读性和性能
result = df.query('age > 25 and salary < 60000')
```

### 2. 内存管理

```python
# 查看内存使用
print(df.memory_usage(deep=True))

# 优化数据类型
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df = optimize_dtypes(df)
```

### 3. 链式操作

```python
# 使用方法链提高代码可读性
result = (df
    .query('age > 25')
    .groupby('department')
    .agg({'salary': 'mean'})
    .sort_values('salary', ascending=False)
    .reset_index()
)
```

### 4. 数据验证

```python
# 使用assert进行数据验证
assert df['age'].min() >= 0, "年龄不能为负"
assert df.duplicated().sum() == 0, "存在重复数据"
assert df.isnull().sum().sum() == 0, "存在缺失值"
```

---

## ⚠️ 常见问题和注意事项

### 问题1: SettingWithCopyWarning

```python
# ❌ 会触发警告
subset = df[df['age'] > 25]
subset['new_col'] = 1  # 警告!

# ✅ 正确做法
subset = df[df['age'] > 25].copy()
subset['new_col'] = 1
```

### 问题2: 链式索引

```python
# ❌ 链式索引(可能不工作)
df[df['A'] > 0]['B'] = 1

# ✅ 使用loc
df.loc[df['A'] > 0, 'B'] = 1
```

### 问题3: 内存溢出

```python
# 对于大文件,使用分块读取
chunks = []
for chunk in pd.read_csv('large.csv', chunksize=10000):
    processed = process_chunk(chunk)
    chunks.append(processed)
result = pd.concat(chunks)
```

### 问题4: 日期解析慢

```python
# ❌ 自动推断日期(慢)
df = pd.read_csv('data.csv', parse_dates=True)

# ✅ 明确指定日期列和格式
df = pd.read_csv('data.csv',
                 parse_dates=['date_col'],
                 date_format='%Y-%m-%d')
```

### 问题5: 性能对比

当数据量 > 1GB 或需要极致性能时:
- 考虑 **Polars** (30x性能提升)
- 考虑 **Dask** (分布式处理)
- 考虑 **DuckDB** (SQL分析)

---

## 📖 进阶资源

### 官方文档
- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [10分钟入门Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)

### 推荐书籍
- "Python for Data Analysis" by Wes McKinney (Pandas作者)
- "Pandas Cookbook" by Matt Harrison

### 在线教程
- [Kaggle Learn - Pandas](https://www.kaggle.com/learn/pandas)
- [DataCamp - Pandas Courses](https://www.datacamp.com/courses/pandas-foundations)

---

## 🔗 相关Skills

- **numpy-skill**: Pandas的底层依赖
- **polars-skill**: 高性能替代方案
- **matplotlib-skill**: 数据可视化
- **jupyter-skill**: 交互式开发环境

---

**最后更新**: 2026-01-22
**版本**: 2.2.x
