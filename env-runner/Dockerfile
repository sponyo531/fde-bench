# FDE-bench 一体化 runner 镜像
#
# 目标：一个镜像跑通发布 case、全部脚手架、以及 harness 自身。
# **运行时不装任何东西**——不 pip install、不挂共享目录的 pip_packages、
# 不依赖 PYTHONPATH 拼接。
#
# 为什么不复用已有的 runner 镜像：
#   ① 它只有 6/22 个科学计算包，缺 ortools / pulp / cvxpy / highspy / torch /
#      xgboost 等 —— 这些是组合优化和 ML case 的核心求解器；
#   ② 它只有 opencode + claude-code 两个脚手架，缺 codex / gemini / kimi / openhands；
#   ③ webagent 靠 PYTHONPATH 挂共享目录补包，而那是可变共享状态——
#      实测一次 `pip install --upgrade` 顺带升级 + 超时中断，就让 numpy 的
#      openblas .so 半写、numpy/cv2/matplotlib 集体 import 失败，而三个 submit
#      脚本都挂着它。依赖必须固化进镜像。
#
# 构建示例（可按所在网络替换镜像源）：
#   cd FDE-bench && docker build --network=host \
#     --build-arg PIP_INDEX_URL=https://pypi.org/simple \
#     -t fde-bench-runner:v1 -f env-runner/Dockerfile env-runner \
#     > /tmp/build.log 2>&1; echo $?
#
# ⚠️ 不要用 `docker build ... | tail -N` 判成功——管道退出码是 tail 的，恒为 0。
#    实测因此误报过一次「镜像建成了」，真实情况是 pip 网络超时挂了。

# 基础镜像使用公开的 ECR 镜像，构建环境也可以通过 FROM 覆盖。
FROM public.ecr.aws/docker/library/python:3.12-slim

# 网络速度取决于部署环境；构建参数允许使用本地或组织镜像。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── 系统层：只装运行时库，**不装编译器** ──────────────────────────────────
# 这一条照发布 case Dockerfile 的既有做法：全部 pip 依赖都有 manylinux
# wheel（含 ortools / cvxpy / highspy / shapely / torch），不需要 build-essential
# 与任何 -dev 包。实测那些 case 镜像 99 秒建完，而我第一版装了 build-essential
# （147 个 apt 包 / 645 MB / 官方源 91 kB/s），apt 那一层就跑了二十多分钟还没完。
#
#   libgomp1            lightgbm / xgboost / ortools 的 OpenMP 运行时
#                       （缺它 import 报 OSError: libgomp.so.1）
#   libgl1 + libglib2.0-0  opencv（029 的 pyquaticus import cv2）
#   libsdl2-2.0-0       pygame（029）
#   curl + ca-certificates  装 node 用，装完即可（不留编译链）
#   ripgrep                 OpenCode glob 工具的本地搜索后端；预装避免运行时下载
#
# apt 源可通过 APT_MIRROR 覆盖，以适配不同部署环境。
ARG APT_MIRROR=mirrors.huaweicloud.com
RUN sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 libsdl2-2.0-0 curl ca-certificates ripgrep \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

# ── 四个 npm 脚手架 ───────────────────────────────────────────────────────
# 与 harness/backends/ 一一对应：opencode / claude_code / codex / gemini / deepseek_harness。
# 另两个走 pip：kimi（PyPI 包 kimi-cli，npm 上的 @moonshot-ai/kimi-cli 不存在，
# 叫 kimi 的是别人 2016 年的动画库）、openhands（独立 venv，见下）。
# opencode / claude-code 版本与既有 runner 对齐（实测 1.16.2 / 2.1.170）——
# 换版本会改变被测 agent 的行为，跨批次分数不可比。
RUN npm install -g --no-fund --no-audit \
        opencode-ai@1.16.2 \
        @anthropic-ai/claude-code@2.1.170 \
        @openai/codex@0.147.0 \
        @google/gemini-cli \
    && npm cache clean --force

# DeepSeek Harness（scaffold=deepseek-harness，配 chat/deepseek-v4-pro）。
# 做法与 Dockerfile.dsh-overlay 完全一致，两个坑见那里的注释：
#   1. dsh 要 Node ≥22（系统 Node 是 20）→ 单独装 /opt/node22，只给 dsh 用
#   2. 不能 npm i -g dsh@alpha.5（子包会解析到 rc.1 混版本）→ 用 dsh/package-lock.json npm ci
ARG NODE22=22.23.2
RUN set -e; cd /opt \
    && python3 -c "import urllib.request; urllib.request.urlretrieve('https://npmmirror.com/mirrors/node/v${NODE22}/node-v${NODE22}-linux-x64.tar.gz','node22.tar.gz')" \
    && mkdir -p /opt/node22 && tar -xzf node22.tar.gz -C /opt/node22 --strip-components=1 && rm node22.tar.gz
COPY dsh/package.json dsh/package-lock.json /opt/dsh/
RUN set -e; cd /opt/dsh \
    && PATH=/opt/node22/bin:$PATH npm ci --no-fund --no-audit --registry=https://registry.npmmirror.com \
    && PATH=/opt/node22/bin:$PATH npm cache clean --force \
    && printf '#!/bin/sh\nexec /opt/node22/bin/node /opt/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js "$@"\n' > /usr/local/bin/dsh \
    && chmod +x /usr/local/bin/dsh && dsh --version

ARG PIP_INDEX_URL=https://pypi.org/simple

# ── torch 单独一层（最重的一步，约 2.5 GB CUDA 轮子）────────────────────────
# 单独成层：超时重试能从缓存续上，不必重下 2.5 GB。
# ⚠️ 必须 CUDA 版：在有 GPU 的运行环境中，装 CPU 版会让 torch 静默退回 CPU，表面看是「模型训得慢/
#    效果差」，极难排查。
RUN pip install --no-cache-dir --retries 10 --timeout 180 \
        --index-url $PIP_INDEX_URL \
        torch==2.6.0

# ── 全部 22 个 case 依赖，版本锁死 ────────────────────────────────────────
# 版本取自发布 case 的 environment/Dockerfile（各依赖组的并集）。
# 锁定的理由：quality 是相对 baseline 的比值，求解器版本漂移会让分数不可比。
# opencv 必须 ≤4.10.x —— 4.11+ 要求 numpy>=2，与锁定的 numpy 1.26.4 冲突
# （029 原先钉 4.13.0.92，pip 判 ResolutionImpossible，那个镜像从来没建成过）。
RUN pip install --no-cache-dir --retries 10 --timeout 180 \
        --index-url $PIP_INDEX_URL \
        numpy==1.26.4 \
        pandas==2.3.2 \
        scipy==1.15.3 \
        scikit-learn==1.7.2 \
        matplotlib==3.10.7 \
        openpyxl==3.1.5 \
        xlrd==2.0.2 \
        ortools==9.11.4210 \
        pulp==2.9.0 \
        cvxpy==1.6.5 \
        highspy==1.11.0 \
        networkx==3.4.2 \
        shapely==2.1.2 \
        xgboost==3.1.1 \
        lightgbm==4.1.0 \
        statsmodels==0.14.6 \
        numba==0.61.2 \
        gymnasium==1.3.0 \
        pettingzoo==1.26.1 \
        pygame==2.6.1 \
        opencv-python==4.10.0.84

# ── agent 侧常用件（按 data/ 的真实格式与任务类型反推）────────────────────
# 这一层不是 case Dockerfile 里有的，是这个统一镜像额外补的。依据是实测
# data/ 目录的文件格式分布与任务类型，不是凭感觉列清单：
#
#   pyarrow      685 个 .parquet —— pandas.read_parquet 没有它直接报
#                ImportError: Unable to find a usable engine
#   meshio       187 个 .vtu（002 的 CFD 表面网格）；否则要手写 VTK XML 解析
#   numpy-stl    44 个 .stl（056 的冒口几何）
#   chardet      057 / 058 的 CSV 是 GBK；不猜编码会 UnicodeDecodeError
#   openpyxl 已有（写 xlsx 用 xlsxwriter 更顺手，一并带上）
#
# 另外几个是这类任务的常用件，装了成本很低、缺了 agent 容易卡住：
#   pyomo        建模层，写 MIP 比直接调 ortools API 顺手
#   deap         遗传/进化算法，连续与组合寻优 case 常用
#   optuna       超参搜索
#   catboost     xgboost / lightgbm 的同侪，树模型三件套缺一不可比
#   imbalanced-learn  不平衡数据任务常用
#   geopy        大圆距离；081/098/100 的 evaluator 自己手算 haversine，
#                但 agent 未必想手写
#   tqdm / requests / pyyaml / joblib  通用件
RUN pip install --no-cache-dir --retries 10 --timeout 180 \
        --index-url $PIP_INDEX_URL \
        pyarrow \
        meshio \
        numpy-stl \
        chardet \
        xlsxwriter \
        pyomo \
        deap \
        optuna \
        catboost \
        imbalanced-learn \
        geopy \
        tqdm \
        requests \
        pyyaml \
        joblib

# ── harness 自身的依赖 ────────────────────────────────────────────────────
# extractor_agent.py import claude_agent_sdk；answerer / judge 走 openai 协议；
# config 读 toml（3.12 自带 tomllib，仍装 tomli 兜旧代码路径）。
RUN pip install --no-cache-dir --retries 10 --timeout 180 \
        --index-url $PIP_INDEX_URL \
        claude-agent-sdk anthropic openai httpx tomli anyio \
        kimi-cli==1.49.0

# ── OpenHands：独立 venv ──────────────────────────────────────────────────
# 为何隔离：OpenHands 要 rich>=14，而它跟主环境其它包的 rich 约束冲突过
# （config.toml [openhands] 的注释记着本机 rich 13.7.1 被 17 个包依赖）。
# 镜像里主环境未必有那 17 个包，但保持与 config.toml 的 `venv = ".venv-openhands"`
# 一致，harness 才找得到——路径由 DELIVER_OPENHANDS_VENV 指过来。
# ⚠️ 版本必须锁 1.6.0：driver.py 走 legacy API（openhands.core.config / core.main.run_controller），
# 1.11 已删除 openhands.core，import 即死，pod 里每个 OpenHands run 都 degraded（2026-09-04 实测）。
# 升到新版 SDK 要重写 driver 的运行/事件/澄清回调/token 采集链路，另开任务。
RUN python3 -m venv --system-site-packages /opt/venv-openhands \
    && /opt/venv-openhands/bin/pip install --no-cache-dir --retries 10 --timeout 180 \
        --index-url $PIP_INDEX_URL \
        'rich>=14' openhands-ai==1.6.0 litellm==1.77.7
ENV DELIVER_OPENHANDS_VENV=/opt/venv-openhands

# ── 运行期 pip 源 ─────────────────────────────────────────────────────────
# agent 若要装未预装的库，直接 pip install 即可。写进 pip.conf 而非依赖命令行
# 参数——agent 不该知道镜像源地址，那是环境的事。网络受限时若不配，agent 装包
# 会超时并退化成朴素解法，表现为「不会用某求解器」，实则是环境问题。
RUN { echo "[global]"; \
      echo "index-url = $PIP_INDEX_URL"; \
      echo "timeout = 120"; } > /etc/pip.conf

# ── 构建期自检：一个不过就不出镜像 ────────────────────────────────────────
# 科学栈按 case 的实际 import 名核对（注意 sklearn/cv2/pywraplp 的名字与包名不同）。
RUN python3 - <<'PY'
import importlib, sys
mods = ['numpy','pandas','scipy','sklearn','matplotlib','openpyxl','xlrd',
        'ortools.linear_solver.pywraplp','pulp','cvxpy','highspy','networkx',
        'shapely','torch','xgboost','lightgbm','statsmodels.api','numba',
        'gymnasium','pettingzoo','pygame','cv2',
        'claude_agent_sdk','anthropic','openai','httpx',
        # agent 侧常用件
        'pyarrow','meshio','stl','chardet','xlsxwriter','pyomo','deap',
        'optuna','catboost','imblearn','geopy','tqdm','requests','yaml','joblib']
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
if bad:
    print("SELFCHECK FAILED:", *bad, sep="\n  ")
    sys.exit(1)
import numpy, pandas, torch
print(f"deps OK — numpy {numpy.__version__} pandas {pandas.__version__} "
      f"torch {torch.__version__} cuda_built={torch.version.cuda}")
PY

# 六个脚手架的可执行都要在 PATH 上；openhands 验 venv 里能 import。
RUN set -e; \
    for b in opencode claude codex gemini kimi; do \
      command -v "$b" >/dev/null || { echo "SELFCHECK FAILED: 缺 $b"; exit 1; }; \
      echo "  $b -> $(command -v $b)"; \
    done; \
    command -v rg >/dev/null || { echo "SELFCHECK FAILED: 缺 rg (ripgrep)"; exit 1; }; \
    rg --version | head -1; \
    /opt/venv-openhands/bin/python -c "import openhands; from litellm.main import responses_api_bridge_check; assert responses_api_bridge_check('responses/gpt-6-astra','openai')[0].get('mode') == 'responses'; print('  openhands Responses bridge OK')"; \
    echo "scaffolds OK (6/6)"

# ── opencode 的 provider SDK 与模型元数据（**不含任何模型配置**）─────────────
#
# 这一层与「配置」的分界，是这个镜像最容易搞混的地方，写清楚：
#
#   进镜像（本层）      provider 的 SDK 适配器 + models.dev 的公共模型清单
#                       —— 纯运行依赖，没有 key、没有 baseURL、没有模型白名单
#   留在运行环境（不进镜像） opencode/opencode.jsonc
#                       —— provider 定义、白名单、{env:} 占位符
#
# 这么分的理由：**加新模型不该要求重建 10 GB 镜像**。加模型只改那份 jsonc，
# 提交流程会把配置 stage 成严格 JSON 放进 run 目录，harness 从那儿继承。
#
# ① SDK：provider 的 "npm" 字段要求包真实存在。隔离配置目录里没有
#    node_modules 时 opencode 会尝试联网安装，离线/受限网络下 provider 加载
#    失败，而 CLI 只报一句含混的 "Session not found"。
#    装到 /root/.config/opencode 是因为 harness 的
#    `environment.py::_seed_provider_sdk()` 从 `Path.home()/".config"/"opencode"`
#    搬 node_modules —— 那是**硬编码的 HOME**，不看 XDG_CONFIG_HOME。
#    于是 HOME 侧供 SDK、XDG_CONFIG_HOME 侧供配置，两者各管一头。
#
# ② models.json：opencode 启动时拉 models.dev 的模型清单，拉不到就退出。
#    烤进 /root/.cache/opencode/ 是 `_seed_models_cache()` 的兜底查找位置
#    （它先看 $XDG_CACHE_HOME，再回落 ~/.cache），这样 pod 里通不通外网都不影响。
#    这是公共模型元数据，不是配置。
RUN npm install -g --no-fund --no-audit \
        @ai-sdk/openai-compatible @ai-sdk/openai@2.0.122 \
    && mkdir -p /root/.config/opencode/node_modules \
    && cp -a "$(npm root -g)/@ai-sdk" /root/.config/opencode/node_modules/ \
    && npm cache clean --force
COPY models.json /root/.cache/opencode/models.json

# 自检：这两样缺任何一个，症状都是「CLI exit 0、零产物」，事后极难归因，
# 所以在构建期就判死，不让这种镜像出厂。
RUN set -e; \
    test -d /root/.config/opencode/node_modules/@ai-sdk/openai-compatible \
      || { echo "SELFCHECK FAILED: 缺 @ai-sdk/openai-compatible"; exit 1; }; \
    test -d /root/.config/opencode/node_modules/@ai-sdk/openai \
      || { echo "SELFCHECK FAILED: 缺 @ai-sdk/openai"; exit 1; }; \
    python3 -c "import json,sys; d=json.load(open('/root/.cache/opencode/models.json')); \
print(f'  models.json OK — {len(d)} 家厂商')"; \
    test ! -e /root/.config/opencode/opencode.json \
      || { echo "SELFCHECK FAILED: 镜像里出现了 opencode.json —— 模型配置不该进镜像"; exit 1; }; \
    echo "opencode SDK OK（配置不在镜像内，按设计）"

WORKDIR /workspace
CMD ["/bin/bash"]
