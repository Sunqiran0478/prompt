# Security Policy

## Supported versions

PromptOS 目前处于 Alpha 阶段，仅最新提交和最新发布版本接收安全修复。

## Reporting a vulnerability

请不要为未修复的漏洞创建公开 Issue。使用 GitHub 仓库的 **Security →
Report a vulnerability** 私下报告；如果仓库尚未启用私密漏洞报告，请联系
仓库维护者并仅提供最少必要信息。

报告应包含受影响版本、复现步骤、影响范围和建议的缓解方式。请勿附带真实
API Key、生产数据或个人信息。维护者确认问题并准备修复前，请避免公开披露。

## Credential handling

PromptOS 只从运行时环境变量读取网关凭据。任务配置、缓存、运行产物、测试
夹具和日志都不应包含认证头、API Key 或 token。
