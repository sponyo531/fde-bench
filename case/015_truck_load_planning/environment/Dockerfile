# 015_truck_load_planning — 运行环境
# task_type: Combinatorial Optimization
#
# 组合优化：装 ortools / pulp / cvxpy / highspy / networkx / shapely
#
# 构建（context 为本目录）：
# docker build --network=host -t fde-bench:015_truck_load_planning .
#
# 必须加 --network=host：默认 bridge 网络的 DNS 解析不到内网 PyPI 源，
# 会退回上游（约 5 KB/s）导致构建超时。实测本 case 加上后 99 秒完成。
#
# 依赖按 task_type 分组，而非全量塞入：给 46 个组合优化 case 装 2GB 的 torch
# 没有意义。但也不按本 case evaluator 的 import 裁剪——evaluator 只用
# pandas/numpy，而 agent 选哪个求解器不受约束，那正是被测能力的一部分。
#
# 版本锁定至产出参考基线时的环境；漂移会让 quality 分数与 baseline 不可比。
#
# 未预装的库，agent 可自行 pip 安装（容器需联网）。此类安装会被记录，
# 用于识别「环境漂移」的 run。
#
# base 镜像用 public.ecr.aws 而非 Docker Hub：本环境 registry-1.docker.io 的
# DNS 被投毒（解析到无关的 IPv6 段）不可达，ECR 镜像内容与官方一致。

FROM public.ecr.aws/docker/library/python:3.12-slim

# 全部依赖均有 manylinux wheel（含 ortools / cvxpy / highspy / shapely），
# 无需编译器与 -dev 系统库——省掉 apt 一大步，构建快得多。

ENV PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PIP_NO_CACHE_DIR=1

# PyPI 源可覆盖。默认公网；网络受限的环境传入自己的镜像源即可：
# docker build --build-arg PIP_INDEX_URL=<你的镜像源> \
#--build-arg PIP_TRUSTED_HOST=<host> --network=host .
# 该值同时写入镜像内的 /etc/pip.conf，供 agent 运行期安装缺失的包。
ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_TRUSTED_HOST=pypi.org

RUN pip install --no-cache-dir \
--index-url $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST \
numpy==1.26.4 \
pandas==2.3.2 \
scipy==1.15.3 \
scikit-learn==1.7.2 \
matplotlib==3.10.7 \
openpyxl==3.1.5 \
ortools==9.14.6206 \
pulp==3.3.0 \
cvxpy==1.7.3 \
highspy==1.13.1 \
networkx==3.4.2 \
shapely==2.1.2

# 运行期 pip 源：agent 若需要未预装的库，可直接 pip install。
# 写进 pip.conf 而非依赖命令行参数——agent 不该知道镜像源地址，那是环境的事。
# 网络受限时若不配，agent 装包会超时并退化成朴素解法，表现为「不会用某求解器」，
# 实则是环境问题。
RUN { echo "[global]"; echo "index-url = $PIP_INDEX_URL"; echo "trusted-host = $PIP_TRUSTED_HOST"; } > /etc/pip.conf

WORKDIR /workspace

# 自检：缺任何一个即构建失败，不留到运行期变成静默 fallback
RUN python -c "import cvxpy, highspy, matplotlib, networkx, numpy, openpyxl, ortools, pandas, pulp, scipy, shapely, sklearn; print('environment OK')"
