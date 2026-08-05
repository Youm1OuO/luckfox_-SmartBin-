// ============================================================================
//  cloud_http.cc
//  POSIX HTTP transport used by CloudUploader
// ============================================================================
#include "cloud_uploader.h"

#include <cstdio>
#include <cstring>

#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>

namespace fridge {

// ---------------------------------------------------------------------------
//  HTTP POST（支持可选的 Authorization 请求头）
// ---------------------------------------------------------------------------
bool CloudUploader::http_post(const std::string& path, const std::string& body,
                              std::string& resp_body, bool use_auth) {
    resp_body.clear();

    // 1) 解析主机：先按点分 IP，失败再 DNS
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);

    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        struct hostent* he = gethostbyname(host.c_str());
        if (!he || !he->h_addr_list[0]) {
            printf("[CLOUD] DNS 解析失败: %s\n", host.c_str());
            return false;
        }
        memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
    }

    // 2) 建 socket + 连接
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { printf("[CLOUD] socket() 失败\n"); return false; }

    // 设置发送/接收超时，避免云端无响应时后台线程卡死
    struct timeval tv;
    tv.tv_sec = 5; tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        printf("[CLOUD] connect %s:%d 失败\n", host.c_str(), port);
        close(fd);
        return false;
    }

    // 3) 拼 HTTP 请求头 + body
    char header[1024];
    if (use_auth && !token.empty()) {
        snprintf(header, sizeof(header),
                 "POST %s HTTP/1.1\r\n"
                 "Host: %s:%d\r\n"
                 "Content-Type: application/json\r\n"
                 "Authorization: Bearer %s\r\n"
                 "Content-Length: %zu\r\n"
                 "Connection: close\r\n"
                 "\r\n",
                 path.c_str(), host.c_str(), port, token.c_str(), body.size());
    } else {
        snprintf(header, sizeof(header),
                 "POST %s HTTP/1.1\r\n"
                 "Host: %s:%d\r\n"
                 "Content-Type: application/json\r\n"
                 "Content-Length: %zu\r\n"
                 "Connection: close\r\n"
                 "\r\n",
                 path.c_str(), host.c_str(), port, body.size());
    }

    std::string req = header;
    req += body;

    // 4) 发送（处理部分写）
    size_t total = 0;
    while (total < req.size()) {
        ssize_t n = send(fd, req.data() + total, req.size() - total, 0);
        if (n <= 0) { printf("[CLOUD] send 失败\n"); close(fd); return false; }
        total += (size_t)n;
    }

    // 5) 接收响应（Connection: close，读到对端关闭为止）
    std::string raw;
    char buf[2048];
    while (true) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n > 0)      raw.append(buf, (size_t)n);
        else            break;   // 0=对端关闭, <0=超时/出错
        if (raw.size() > 1 << 20) break;  // 安全上限 1MB
    }
    close(fd);

    if (raw.empty()) { printf("[CLOUD] 空响应\n"); return false; }

    if (raw.find("HTTP/1.1 2") != 0 && raw.find("HTTP/1.0 2") != 0) {
        size_t line_end = raw.find("\r\n");
        std::string status_line = line_end == std::string::npos ? raw : raw.substr(0, line_end);
        printf("[CLOUD] HTTP 非成功状态: %s\n", status_line.c_str());
        return false;
    }

    // 6) 切出 body（找第一个空行 \r\n\r\n）
    size_t pos = raw.find("\r\n\r\n");
    resp_body = (pos == std::string::npos) ? raw : raw.substr(pos + 4);
    return true;
}

}  // namespace fridge
