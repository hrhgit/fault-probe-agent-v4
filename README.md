# fault-probe-agent-v4

`fault-probe-agent-v4` 是 GOSIM 2026 Agentic Observer 项目中的一个参赛者自有智能体版本。这个独立仓库只包含运行该版本所需的参赛者源码和配置，不包含官方平台仓库、任务卡、评测代码、运行记录、密钥或可选本地模型权重。

## v4 方法

v4 的改动集中在交接记录中设备作业时刻的处理：

1. 提示模型保留记录里的日期和钟点，并用 `tz` 标出原时区，例如 `UTC` 或 `+9`。
2. 程序在 [`src/handover/extract.ts`](src/handover/extract.ts) 校验时区并换算时间；模型不负责做时区算术。
3. 预告的作业时刻作为计划事件处理，不会被“已发生故障不能在未来”这条校正规则移回过去。

主要实现位于 [`src/handover/prompt.ts`](src/handover/prompt.ts) 和 [`src/handover/extract.ts`](src/handover/extract.ts)。其余源码是该方法所需的参赛者自有运行底座。

## 构建和运行

需要 Node.js 22。

```sh
npm ci
npm run build
npm start
```

程序从标准输入读取 JSON Lines，并向标准输出写入响应。使用模型读取交接记录时，需要自行配置模型服务环境变量；本仓库不含 API 密钥。若使用 DeepSeek，程序固定请求 `deepseek-flash`，不能通过环境变量改成其他 DeepSeek 型号。

可选的本地模型权重和 Python 运行环境没有随仓库发布；缺少它们时，这些可选模型会自动关闭，交接时区处理不依赖它们。

## 来源和许可

本仓库收录的是参赛者编写的智能体源码，不是 GOSIM 官方平台或其完整仓库。GOSIM 提供的任务卡、评测代码、示例项目等材料不在本仓库内，仍按官方规则中的 CC BY-NC 4.0 许可使用；本仓库遵循官方要求注明活动来源。参赛者代码采用 MIT License，详见 [`LICENSE`](LICENSE) 和 [`NOTICE.md`](NOTICE.md)。

官方活动：[GOSIM 2026 Agentic Observer Hackathon](https://create.gosim.org/survey26/)

官方规则：[中文规则](https://github.com/gosimfoundation/hackathon-survey26/blob/main/web/src/content/rules.zh.md)
