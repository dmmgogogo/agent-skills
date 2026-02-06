# FastAPI Skill

## 📚 工具简介

**FastAPI** 是一个现代、高性能的Python Web框架,用于构建API。它基于标准Python类型提示,自动生成API文档。

### 核心特性
- **极快性能**: 与NodeJS和Go相当
- **自动文档**: Swagger UI和ReDoc
- **类型验证**: 基于Pydantic
- **异步支持**: 原生async/await
- **依赖注入**: 优雅的依赖管理
- **安全性**: OAuth2, JWT开箱即用

### GitHub信息
- **Stars**: 94,000+ (增长最快的Python Web框架)
- **增长率**: 38%年增长
- **仓库**: https://github.com/fastapi/fastapi
- **官方文档**: https://fastapi.tiangolo.com/

### 适用场景
✅ RESTful API开发
✅ 微服务架构
✅ 机器学习模型服务化
✅ 实时应用(WebSocket)
✅ 高性能后端服务

---

## 🔧 安装和配置

### 基础安装

```bash
# 安装FastAPI
pip install fastapi --break-system-packages

# 安装ASGI服务器(生产环境)
pip install "uvicorn[standard]" --break-system-packages

# 完整安装(包含所有可选依赖)
pip install "fastapi[all]" --break-system-packages
```

### 常用依赖

```bash
# 数据库支持
pip install sqlalchemy databases asyncpg --break-system-packages

# 认证
pip install python-jose[cryptography] passlib[bcrypt] --break-system-packages

# 文件上传
pip install python-multipart --break-system-packages

# 测试
pip install pytest httpx --break-system-packages
```

---

## 💻 代码示例

### 1. Hello World

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# 运行: uvicorn main:app --reload
# 访问: http://localhost:8000
# 文档: http://localhost:8000/docs
```

### 2. 请求体验证(Pydantic)

```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=0, le=120)
    is_active: bool = True
    bio: Optional[str] = None

@app.post("/users/")
async def create_user(user: User):
    return {"user": user, "message": "User created"}

# 请求示例:
# POST /users/
# {
#   "username": "johndoe",
#   "email": "john@example.com",
#   "age": 30
# }
```

### 3. 路径参数和查询参数

```python
from enum import Enum
from typing import List

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name, "message": "Deep Learning FTW!"}

@app.get("/search/")
async def search_items(
    q: str,
    skip: int = 0,
    limit: int = 10,
    tags: List[str] = []
):
    return {
        "query": q,
        "skip": skip,
        "limit": limit,
        "tags": tags
    }
# 访问: /search/?q=python&skip=0&limit=20&tags=web&tags=api
```

### 4. 异步数据库操作

```python
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

DATABASE_URL = "postgresql://user:password@localhost/dbname"
database = Database(DATABASE_URL)
metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(50)),
    Column("email", String(100))
)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/users/{user_id}")
async def read_user(user_id: int):
    query = users.select().where(users.c.id == user_id)
    return await database.fetch_one(query)

@app.post("/users/")
async def create_user(name: str, email: str):
    query = users.insert().values(name=name, email=email)
    user_id = await database.execute(query)
    return {"id": user_id, "name": name, "email": email}
```

### 5. 认证和授权(JWT)

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # 验证用户(示例)
    if form_data.username == "test" and form_data.password == "test":
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Incorrect credentials")

@app.get("/protected")
async def protected_route(username: str = Depends(verify_token)):
    return {"message": f"Hello {username}"}
```

### 6. 文件上传

```python
from fastapi import File, UploadFile
from typing import List

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents)
    }

@app.post("/upload-multiple/")
async def upload_multiple(files: List[UploadFile] = File(...)):
    return {
        "filenames": [file.filename for file in files]
    }
```

### 7. 后台任务

```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(f"{message}\n")

@app.post("/send-notification/{email}")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    return {"message": "Notification sent"}
```

### 8. WebSocket

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message received: {data}")
```

---

## 🎯 最佳实践

### 1. 项目结构

```
my_project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI应用入口
│   ├── models.py        # Pydantic模型
│   ├── schemas.py       # 数据库模型
│   ├── crud.py          # 数据库操作
│   ├── dependencies.py  # 依赖注入
│   └── routers/         # 路由模块
│       ├── __init__.py
│       ├── users.py
│       └── items.py
├── tests/
├── requirements.txt
└── .env
```

### 2. 使用路由器组织代码

```python
# routers/users.py
from fastapi import APIRouter

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("/")
async def get_users():
    return [{"username": "user1"}]

# main.py
from fastapi import FastAPI
from .routers import users

app = FastAPI()
app.include_router(users.router)
```

### 3. 依赖注入

```python
from fastapi import Depends

def common_parameters(q: str = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

# 数据库会话依赖
async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
async def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

### 4. 错误处理

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "Custom header"}
        )
    return items[item_id]

# 自定义异常处理器
from fastapi.responses import JSONResponse

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name}"}
    )
```

### 5. CORS配置

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## ⚠️ 常见问题和注意事项

### 问题1: async vs sync

```python
# 使用async当:
# - 使用异步库(databases, httpx)
# - I/O密集型操作
async def read_users():
    return await database.fetch_all(query)

# 使用sync当:
# - CPU密集型操作
# - 使用同步库(sqlalchemy ORM)
def compute_heavy():
    return complex_calculation()
```

### 问题2: Pydantic配置

```python
class UserCreate(BaseModel):
    username: str
    password: str

    class Config:
        # 允许ORM对象转换
        orm_mode = True
        # 字段示例(文档中显示)
        schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "secret"
            }
        }
```

### 问题3: 生产部署

```bash
# 使用Gunicorn + Uvicorn
pip install gunicorn --break-system-packages

# 启动命令
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 📖 进阶资源

- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [FastAPI GitHub仓库](https://github.com/fastapi/fastapi)
- [Awesome FastAPI](https://github.com/mjhea0/awesome-fastapi)

---

## 🔗 相关Skills

- **pydantic-skill**: 数据验证
- **sqlalchemy-skill**: ORM
- **docker-skill**: 容器化部署
- **pytest-skill**: API测试

---

**最后更新**: 2026-01-22
