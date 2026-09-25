# cases runner — 统一实验入口（P4 实现，规格见蓝图 §4）

cases run|agg|status|canary|verify|doctor
纪律接合：prereg 硬闸（scored 无冻结 registry 拒跑）/ Holm family / append-only ledger / 停止准则四条 / 七 cap。
收编方式：src/cases/runner/adapters/legacy/<line>.py 适配层翻译现役 v8_run_*.py 的 argparse 语义，零侵入。
