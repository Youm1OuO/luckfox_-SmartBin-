# 使用 Ubuntu 22.04 作为基础镜像，这是 Buildroot 官方推荐的编译环境之一
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/ubuntu:22.04

# 设置环境变量，使 apt 安装时无需交互式确认
ENV DEBIAN_FRONTEND=noninteractive

# 更新软件源并安装 Buildroot 所需的所有依赖包
# 这些包涵盖了编译、打包、文件系统工具等各个方面
RUN apt-get update && \
    apt-get install -y \
    # 基础编译工具
    build-essential \
    # 版本控制和文件处理
    git sed make binutils gcc g++ bash patch gzip bzip2 perl \
    tar cpio unzip rsync file bc findutils wget \
    # 配置界面依赖 (menuconfig)
    libncurses-dev \
    # 其他常见工具
    vim less && \
    # 清理 apt 缓存以减小镜像体积
    rm -rf /var/lib/apt/lists/*

# 设置容器启动时的默认工作目录
# 我们将在运行容器时将 Windows 上的项目目录挂载到这个路径
WORKDIR /buildroot

# 容器启动后默认执行 bash 终端，方便手动输入编译命令
CMD ["/bin/bash"]