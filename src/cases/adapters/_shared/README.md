# _shared — E4 通用工具

## truncation_gold_sample.py（E4-0.9 截断金样本, 预注册 tripwire）

- 输入: `work/G_generalists/truncation_gold/{system}_gold_prompts.jsonl` — 每行
  `{"system", "tag", "text"}`; **text 必须是该系统管线会真实发出的提示词**（由各系统
  自己的 prompt 构建路径 + 原生示例输入提取; 每系统 20 条）。
- 输出: `{system}_gold_result.json` — `rate_finish_ne_stop` + histogram + rows。
- Tripwire（预注册）: finish_reason != "stop" > 5% → 该系统 GLM-backed scored 取消,
  native-model-only + DECISION（R2 §1.2/§2.2; manifest.truncation_audit）。
- 并发: 顺序发送（并发 1）+ 1s 间隔; GLM 全局并发上限 10, E 线批在跑时 ≤3。
- 提取状态: S2 需 s2_bfts env（torch import 链）; G4 需 requirements 装齐; S3 需
  openai/pandas; G3 需其原生依赖。env 就绪后逐系统提取 → 跑样 → 记 manifest。

## apply_patches 约定（四系统统一）

pin 静止态逐字节不变（S2: digest 验证; S3/G4: git status）; patch 在 run-time 应用/
还原; 状态写 `work/G_generalists/runs/<system>/patch_state.json`。
