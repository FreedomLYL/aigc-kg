# AIGC 课程知识图谱 · 云端部署说明

本目录是该系统用于 **云端长期可访问部署** 的版本：
- 不含本地的 Windows 启动脚本、数据库文件和密钥；
- 采用单文件前端（`static/index.html` 全内联），无需额外静态托管；
- 免费云实例没有持久磁盘，因此靠 `SEED_ON_START=1` 在每次启动、库为空时自动从 `seed/seed_data.json` 恢复 9 门课程图谱和演示账号，保证专家随时打开都能看到完整演示数据。

## 一、推到 GitHub（准备一个远程仓库）

在 GitHub 网页新建一个空仓库（例如 `aigc-kg`，不需要勾选 README），然后在项目目录执行：

```bash
git init
git add .
git commit -m "AIGC 课程知识图谱 云端部署版"
git branch -M main
git remote add origin https://github.com/<你的用户名>/aigc-kg.git
git push -u origin main
```

## 二、在 Render 上创建 Web Service（约 3 分钟，免费）

1. 打开 https://render.com ，用 GitHub 账号注册/登录（免费会送你足够的额度）。
2. 点 **New + → Web Service**，选择刚推送的 `aigc-kg` 仓库（会自动连接 GitHub，需授权一次）。
3. 按以下填好后点 **Create Web Service**：
   - **Name**: `aigc-kg`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: **Free**
4. 在 **Environment** 里添加环境变量：
   - `SEED_ON_START` = `1`（必填，用于恢复演示数据）
   - 推荐再加（真实 AI）：`DEEPSEEK_API_KEY`、`LLM_PROVIDER=deepseek`
5. 点击 **Create Web Service**，等待首屏构建完成（约 2~4 分钟），之后会自动得到一个固定域名：

   比如 `https://aigc-kg.onrender.com`

6. 打开该网址即可看到系统首页。用它注册/登录演示账号即可进入。（预置为空的库也会自动恢复演示数据；免费实例约 15 分钟无访问会进入休眠，再次打开会先唤醒几秒，属正常现象。）

## 三、（可选）启用真正的数据持久化

默认用的是仓库内置 SQLite 演示库，云端重启会自动重建演示数据，但不会保留专家/评委在上面的**新增改动**。如果希望改动也长期保留：

1. 到 https://supabase.com 免费创建一个数据库，拿到 `DATABASE_URL`（`postgresql://...`）；
2. 在 Render 的 Environment 里加 `DATABASE_URL` 即可（系统内置 `psycopg`，会自动切换为 PostgreSQL 持久存储）。

## 四、其他可选平台

本项目同样适配 **Railway**、**Zeabur**：把 Start Command 和上述两个环境变量配置到对应平台的 Web Service 即可，域名分别是 `*.up.railway.app`、`*.zeabur.app`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `render.yaml` | Render 声明式配置（可直接 “Blueprint” 方式导入） |
| `Procfile` | 网页进程启动命令，供 Render/Railway 识别 |
| `.env.example` | 环境变量模板（含密钥位与说明） |
| `.gitignore` | 排除本地上传的数据库、密钥与缓存 |
| `app/ static/ samples/ seed/` | 服务端、前端、真实课程讲义知识库、种子数据 |