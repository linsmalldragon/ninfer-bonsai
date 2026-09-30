# ninfer-bonsai

> **English TL;DR** — Docker deployment that serves the
> [Ternary-Bonsai-2-27B NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3)
> ternary 27B model (Qwen3.8-27B base, text + vision) on **one RTX 5090** with the
> [NInfer-all](https://github.com/iamwavecut/ninfer-all) engine: 880K-token context, DFlash2
> speculative decoding, OpenAI-compatible API.

在单张 RTX 5090(32 GiB)上,用 [NInfer-all](https://github.com/iamwavecut/ninfer-all) 引擎以 Docker 方式服务
[Ternary-Bonsai-2-27B NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3)
——PrismML 的三值化 27B 模型(Qwen3.8-27B 底座),单个 8.87 GiB `.ninfer` 文件,**内置视觉塔**。
`run.sh` 固化的部署:880K(901,120 token)上下文 + YaRN、rk4v4 KV、DFlash2 投机解码、视觉 overlay
驻留、32k 默认输出上限,OpenAI 兼容 API。

模型本身不进 git;NInfer-all 源码(第三方)也不进 git,README 里写清了从哪里取。

## 目录结构

```
Dockerfile   # 两阶段构建:NInfer-all@2172a598 源码(sm_120a 编译)→ 运行时镜像(entrypoint + 全部 env 默认值)
run.sh       # 一键启动:模型硬链接、880K 部署默认值、unless-stopped 重启策略
bench/       # 压测脚本与结果(C=1/2/4 全上下文、C=4×128k KV 超订、诊断工具、引擎日志摘录)
model/       # (git-ignored) HF 缓存中 .ninfer 的硬链接目录,只读挂载进容器
src/         # (git-ignored) ninfer-all @ 2172a598 的 git 检出,作为 docker build 的 context
```

## 部署规格(本仓库默认)

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | Ternary-Bonsai-2-27B NInfer-v3 | 8.87 GiB,含视觉塔;`n_params≈30.4B`,词表 151,665 |
| 上下文窗口 | **901,120 tokens(880K)+ `--rope-yarn`** | 质量边界实测,见下 |
| KV 编码 | `rk4v4`(镜像默认) | 17.5 KiB/token(16 层 full-attention × 4 KV 头) |
| GDN 状态 | fp16(镜像默认) | 48 层 GDN 线性注意力 |
| 投机解码 | `dflash2`,5 drafts(镜像默认) | 接受率随上下文深度/并发压力下降(33–68%) |
| 视觉 | `--vision --vision-residency overlay --vision-max-merged 12288` | 塔驻留 host 内存,显存成本仅 0.03 GiB;每次编码临时借用输出头/词表/drafter 的显存,用后归还 |
| 默认输出上限 | `--default-max-tokens 32768` | 仅约束**未带** `max_tokens` 的请求;显式 `max_tokens`(含 `-1` 无上限)照单生效 |
| 显存占用 | 运行时 ~15.9 GiB,空闲 ~7.4 GiB | KV 池 14,080/14,080 pages;262K 窗口时 ~13.7 GiB |
| 并发 | 引擎默认单 slot(`/props` `total_slots=1`) | 压测结论 C=1 聚合吞吐最大,见性能参考 |
| API | `http://<host>:8080/v1` | `/chat/completions`、`/responses`、`/v1/messages`(Anthropic 风格)、`/v1/models`、`/props`、`/metrics`、`/slots` |

**为什么是 880K 而不是 1M:** 模型卡针测的结论是检索质量按**位置**衰减而不是按窗口大小——
978,944 窗口内 33/66/90% 位置的全部针都能取回;1,048,576 时 90% 位置的针(≈943K)取不回来
(无论是否 `--rope-yarn`)。本部署取 901,120(880×1024,最浅于 881K 已验证可用的边界),
1M 的 ~20K token 余量换来的是确定的检索退化,弃之。想回 256K 全窗口:`MAX_CONTEXT=262144 EXTRA_ARGS=""`
(并去掉 rope-yarn);想开 1M:`MAX_CONTEXT=KV_CAPACITY=1048576` + `EXTRA_ARGS="--rope-yarn"`
(显存够,质量风险自担)。

## 获取模型(不在此仓库)

模型在 HuggingFace 公开仓库(**公开且无门禁,匿名即可下载**,license: apache-2.0):

<https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3>

```bash
# 一次性:安装下载 CLI
pip install -U huggingface_hub

# 方式 1(推荐):下到 HF 缓存默认位置,run.sh 的硬链接逻辑会自动把它接进 ./model/
huggingface-cli download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3
# (新版 huggingface_hub 的 CLI 也写作:  hf download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3)
# 文件落在 ~/.cache/huggingface/hub/models--WaveCut--Ternary-Bonsai-2-27B-NInfer-v3/snapshots/<snap>/

# 方式 2:直接下到 run.sh 的挂载目录(不依赖 HF 缓存布局,机器上没有缓存也能跑)
huggingface-cli download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 --local-dir /root/ninfer-bonsai/model

# 校验(下载的 SHA256SUMS 列出了全部文件的哈希)
cd <下载目录> && sha256sum -c SHA256SUMS
```

产物为 `Ternary-Bonsai-2-27B-ninfer-v3.ninfer`(9,520,051,456 字节,8.87 GiB),同目录附
`SHA256SUMS`、转换报告(`.conversion.json`)与 `NOTICE`。

`run.sh` 从默认 HF 缓存布局取文件并硬链接进 `./model/`(同文件系统,零拷贝):

- 缓存路径写死在 `run.sh` 的 `HF_SNAP`(本机 2026-09 下载的快照目录)。如果你的快照目录名不同,
  要么改 `HF_SNAP`,要么直接把 `.ninfer` 放进 `/root/ninfer-bonsai/model/`——`run.sh` 发现
  HF 路径解析不到时会沿用 `model/` 里已有文件。

模型血缘:prism-ml/Ternary-Bonsai-2-27B-gguf(三值化底座)+ ProCreations 的 MTP 头与 DFlash2
适配器 merge 而成,视觉塔为原版 Qwen3.8-27B mmproj。

## 构建镜像

前置:Linux x86_64 + Docker(BuildKit)+ nvidia-container-toolkit;5090 需要支持 sm_120a /
CUDA 13.1 的 NVIDIA 驱动(用最新 GeForce 驱动)。

```bash
git clone https://github.com/iamwavecut/ninfer-all
cd ninfer-all
git checkout 2172a5985ad761c2223ee60ab196dbdae70a1aec   # 模型卡基准 commit(2026-09-25)

docker build -f /path/to/ninfer-bonsai/Dockerfile \
  -t ninfer-bonsai2-27b:sm120a-2172a598 .
```

- 构建 context 是 **ninfer-all 检出目录**(不是 ninfer-bonsai 目录),Dockerfile 用 `-f` 指定。
- 为什么 `sm_120a`:消费级 Blackwell(5090)没有 tcgen05,三值 `t2_g128_fp16` 路由走
  mma.sync 兼容路径;上游 stock Dockerfile 默认 `CMAKE_CUDA_ARCHITECTURES=86` 会拒绝本产物。
- 基础镜像(nvidia/cuda:13.1.2-devel/runtime,~3.5 GB)走国内 mirror 拉取可能要约 2 小时;
  编译本身在 `--parallel 16`(主机 RAM 上限保护)下约 30–60 分钟。BuildKit 缓存命中后
  重跑只重做变化部分。
- 镜像里已处理一个坑:CUDA runtime 镜像自带的 forward-compat `libcuda` 在 GeForce 卡上会让
  所有 CUDA 调用失败(`cudaErrorCompatNotSupportedOnDevice`),构建时已删除。

## 运行

```bash
./run.sh                      # 后台,GPU0,宿主端口 8080
GPU=1 HOST_PORT=8081 ./run.sh # 换卡 / 换端口
./run.sh --foreground         # 前台附着,Ctrl-C 停
```

`run.sh` 做的事:模型硬链接 → 组装 env → `docker run -d`(带 `--restart unless-stopped`,
宿主机重启后自动拉起)。**所有部署参数都写成了脚本默认值,同时允许环境变量覆盖**
(`${VAR:-...}` 语义):

| 变量 | 默认(本部署) | 说明 |
|---|---|---|
| `MAX_CONTEXT` | `901120` | 上下文窗口(token) |
| `KV_CAPACITY` | `$MAX_CONTEXT` | KV 池容量 |
| `EXTRA_ARGS` | `--rope-yarn --vision --vision-residency overlay --vision-max-merged 12288 --default-max-tokens 32768` | 追加给引擎的参数 |
| `KV_DTYPE` / `SPEC` / `DRAFT_TOKENS` | 不设 → 镜像默认 `rk4v4` / `dflash2` / `5` | 需要时才覆盖 |
| `MODEL_ID` / `HOST` / `PORT` | 不设 → 镜像默认 `bonsai2-27b` / `0.0.0.0` / `8080` | |
| `GPU` / `HOST_PORT` / `IMAGE` / `NAME` | `0` / `8080` / `ninfer-bonsai2-27b:sm120a-2172a598` / `bonsai2-27b-ninfer` | run.sh 自身 |

注意:转发逻辑**只转发非空值**——`SPEC= ./run.sh` 这类空值不会传进容器,entrypoint 会退回
默认 `dflash2`(drafter 没被关掉)。要改 drafter 请给实际值。

启动完成后验证:

```bash
curl -s http://127.0.0.1:8080/props | python3 -m json.tool
# n_ctx=901120, n_predict=32768, total_slots=1, modalities.vision=true
curl -s http://127.0.0.1:8080/v1/models
# context_window=901120, modalities.vision=true, meta.n_ctx_train=262144
```

## API 使用

Base URL `http://<host>:8080/v1`,模型 id `bonsai2-27b`。

```bash
# 文本
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"bonsai2-27b","messages":[{"role":"user","content":"say ok"}]}'

# 视觉:image_url 支持 http(s) URL 或 data URI(base64 内嵌)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "bonsai2-27b",
  "messages": [{"role":"user","content":[
    {"type":"text","text":"Describe this image precisely."},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,<B64>"}}
  ]}]}'
```

- **thinking 默认开启**:响应会带 `reasoning_content`,推理 token 计入 `max_tokens`。
  短答案场景(如视觉描述)小 `max_tokens` 会被推理吃掉,`content` 为空、finish 为 `length`——
  给足(≥4096)或关掉 thinking。
- `/props`:运行时属性(`n_ctx`、`n_predict`、`total_slots`、模态)。`/metrics`:Prometheus 风格统计;
  `/slots`:llama.cpp 风格 lane 表。
- 无音频输入/输出(引擎无音频管线,产物也无音频塔);需要语音就前置 ASR 转文本。

## 性能参考(bench/,单卡 5090,实测)

压测脚本在 `bench/`(`stress.py` 全上下文 C=1/2/4、`c4_128k.py` KV 超订、`diag_c4.py` 客户端诊断),
结果 jsonl 与引擎日志同目录。要点:

| 指标(C=1) | 64K | 128K | 256K |
|---|---:|---:|---:|
| 单流 decode | ~380 tok/s | ~262 tok/s | ~205 tok/s |
| prefill | ~5.7k tok/s(TTFT 11.2s) | ~4.5k tok/s(28.6s) | ~3.2k tok/s(81.4s) |

- **prefill 是单 lane 串行(引擎设计如此)**:调度器同一时刻只放一个请求进 prefill
  (compute-bound 下这是最优),C↑ 不会加速 prefill,只会排队。
- **并发 decode 聚合吞吐崩塌**:C=2 时每 lane 慢约 10×(128K:C=1 238 → C=2 12.4 tok/s),
  KV 池还会先打满 262K pool + 8 GiB host spill。所以聚合输出吞吐最大在 C=1;
  要多用户并发可以开 C=2/4(都测过,稳定),代价是每人更慢。
- 880K/1M 深度:1M 窗口 e2e(990,052-token prompt)TTFT 13m25s(prefill ~1.23k tok/s),
  深度 ~990K 处 decode ~111 tok/s,dflash2 接受率仍 50.6%——投机在极端深度照样工作。
- 视觉请求:media 缓存命中时 TTFT 14.5 ms(99% 命中上下文缓存),decode ~420 tok/s。
- 大请求波之后 **KV 池会保留 ~100% 占用 + host KV pinned ~4.3 GiB**(上下文缓存留给
  前缀复用),下一批请求要竞争——用 `/slots`、`/metrics` 观察,必要时重启清池。

## 排错 / 坑

- **就绪判定**:重启后 HTTP 层比生成管线早 1–2 s 就绪,此时请求可能返回无 `choices` 的错误体
  ——就绪判定用 `/v1/load` 的 `capacity.max_concurrency` + 一次真实生成,别只看 HTTP 200。
- **HF 快照是符号链接链**(snapshot → blobs → sharded store),直接 bind-mount 快照目录进
  容器会 dangling——所以 `run.sh` 硬链接到 `model/` 再挂载,勿绕过。
- **`SPEC=` 空值无效**(见上表),entrypoint 默认会接管。
- 32 GiB 卡上 NInfer(15.9 GiB)与一个 31.5 GiB 的 SGLang 服务无法共存。
- 镜像 tag 里的 `2172a598` 是 NInfer-all commit 前缀;换引擎版本需要按新 commit 重建,
  且模型卡基准数字只对该 commit 有效。

## License

- 引擎 [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all)(Apache-2.0,upstream Neroued/ninfer),
  本部署钉在 [`2172a598`](https://github.com/iamwavecut/ninfer-all/commit/2172a5985ad761c2223ee60ab196dbdae70a1aec)。
- 模型产物 license 见模型卡(apache-2.0);底座见上方"模型血缘"。
- 本仓库脚本(`Dockerfile`、`run.sh`、`bench/`)未附 LICENSE 文件,随仓库分发。
