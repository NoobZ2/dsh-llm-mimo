# dsh-llm-mimo — Xiaomi MiMo 多模态 Provider 插件（DeepSeek Harness）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

为 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 提供 **Xiaomi MiMo 官方 API**（`api.xiaomimimo.com`，OpenAI 兼容）的 LLM provider 路由。注册 `xiaomi-mimo` provider，让 `mimo-v2.5` 的全模态能力（图片 + 文本）接入 harness：

- ✅ **读图**：`read_image` 工具、GUI 上传的图片，都走 MiMo-V2.5 视觉理解（图片描述 / 图表 OCR / 截图识别）
- ✅ **文本**：`mimo-v2.5` / `mimo-v2.5-pro` 文本生成、深度思考（reasoning）、标准函数调用（tool calling）
- ✅ **免重启**：热装配（dsh-super-injector）或持久化装配均可；连接信息（baseURL / 模型清单 / API key）每次请求实时解析，改配置立即生效

---

## 快速开始

### 1. 获取 MiMo API Key

在 [小米 MiMo API 开放平台](https://mimo.mi.com) 注册并创建 API Key。模型清单与限流见官方文档（`mimo-v2.5`：全模态，上下文 1M，最大输出 128K，RPM 100 / TPM 10M）。

### 2. 安装插件

```bash
# 克隆本仓库
git clone https://github.com/NoobZ2/dsh-llm-mimo.git
cd dsh-llm-mimo

# 建立依赖 junction（指向你的 DSH checkout 的 pnpm store）
DSH_CHECKOUT=/path/to/deepseek-harness bash scripts/build.sh
```

然后在运行中的 harness 里热装配（需要已装配 [dsh-super-injector](https://github.com/yjh051108/dsh-super-injector)）：

```
dev_install_package dir=F:/path/to/dsh-llm-mimo
```

或者手动装配（持久化，重启生效）：在 profile 的 `package.json` 中加

```jsonc
{
  "dependencies": { "@dsh-external/dsh-llm-mimo": "link:F:/path/to/dsh-llm-mimo" },
  "dsh": { "profile": { "bundles": [ "...", "@dsh-external/dsh-llm-mimo" ] } }
}
```

### 3. 配置 API Key

把 key 写入 `$DSH_HOME/.credentials.yaml`（与 `DEEPSEEK_API_KEY` 并列）：

```yaml
MIMO_API_KEY: sk-xxxx
```

也可以用环境变量 `MIMO_API_KEY`（或通过 Web GUI 的 Models 设置页写入）。

### 4. 使用

| 场景 | 做法 |
|---|---|
| **会话内读图** | Web GUI 模型选择器切到 **Xiaomi MiMo / MiMo-V2.5**，之后 `read_image`、上传图片直接走 MIMO |
| **子代理读图** | 派发子代理时覆盖路由：`provider: 'xiaomi-mimo', model: 'mimo-v2.5'`（workflow `agent()` 的 provider/model 覆盖；主会话模型不变） |
| **整机默认 MIMO** | `settings.yaml` 的 `agent-default-model` 改为 `provider: xiaomi-mimo, model: mimo-v2.5` |

验证：

```sql
-- 在会话中（模型已切到 MiMo-V2.5）
read_image(file_path: "某张图片.png")
```

能返回图片内容描述即成功。

---

## 模型清单

| Model ID | 模态 | 上下文 | 最大输出 | 说明 |
|---|---|---|---|---|
| `mimo-v2.5` | text + **image** | 1M | 128K | 全模态理解（推荐读图用） |
| `mimo-v2.5-pro` | text | 1M | 128K | 纯文本旗舰 |

模型清单、baseURL、API key 环境变量名、重试策略等均可在 `llm-mimo` 设置节中覆盖：

```yaml
# settings（llm-mimo 节）
llm-mimo:
  apiKeyEnv: MIMO_API_KEY        # 凭据引用
  baseURL: https://api.xiaomimimo.com/v1
  models: [ ... ]                # 自定义模型清单
  maxTokens: 131072
  defaultContextWindow: 1048576
  streamIdleTimeoutMs: 300000
```

环境变量 `MIMO_BASE_URL` 可覆盖端点。

---

## 兼容性说明（踩坑记录）

适配器针对 MiMo API 的实际行为做了以下处理，其他 OpenAI 兼容 provider 也可参考：

1. **Tuple 式 `items` schema 清洗**：MiMo 拒绝 JSON Schema 中 `items` 为数组（tuple）形式（MCP 生成的 `obsidian_read_pdf` / `obsidian_ocr_pdf` 等工具会触发 HTTP 400）。适配器递归清洗，tuple 转 `anyOf`。
2. **流式 tool-call 尾部 `null` delta**：MiMo 在流末尾会发 `id: null` / `name: null` 的增量，若直接覆盖会导致工具名丢失（`unknown tool ""`）。只在收到非空字符串时更新。
3. **reasoningEffort 声明**：harness 会校验请求的 reasoning effort 必须在模型声明列表内（父会话 header 通常带 `max`），而 MiMo 没有 wire 层的 effort 旋钮（深度思考自动开启）。声明 `off/high/max` 三档但不上送该参数。
4. **max_tokens 钳制**：继承的会话上限（如 256K）超过 MiMo 输出上限时会被钳制到模型清单中的 `maxTokens`。
5. **工具结果内的图片提升（hoist）**：MiMo 拒绝 `role: tool` 消息中的图片内容数组。`read_image` 产生的图片块在 tool-result 内，序列化时提升（hoist）到其后第一条 user 消息。
6. **递归图片检测**：harness 的 `contentHasImage` 不递归 tool-result 内部，适配器自带递归检测并始终解析附件服务。

## 目录结构

```
dsh-llm-mimo/
├── lib/index.js        # 插件主体：适配器 + 注册（手写 ESM，无构建步骤）
├── scripts/build.sh    # 建立依赖 junction + 校验
├── demo/read-image.mjs # 独立演示：直接用 MiMo API 读一张图
└── package.json
```

## 演示（不经 harness）

```bash
MIMO_API_KEY=sk-xxxx node demo/read-image.mjs /path/to/image.png
```

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
