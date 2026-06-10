// ============================================================================
//  cloud_uploader.cc
//  端云协同 — 异步出入库事件上报器实现
//
//  实现要点：
//    - HTTP 用最朴素的 POSIX socket 手写 POST，不引第三方库（板子是 uClibc，
//      链接 curl/openssl 很麻烦）。云端用明文 HTTP 即可，演示足够。
//    - JSON 自己手拼，零依赖。
//    - 输出 JSON 严格对应《端侧返回数据格式.txt》出入库事件格式。
//    - 端侧不解析返回（标签识别/配对在后端做），只关心 HTTP 是否连通。
//    - 后台单线程串行发送：事件不是高频，串行够用，避免并发 socket 复杂度。
// ============================================================================
#include "cloud_uploader.h"
#include "fridge_config.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>

#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>

namespace fridge {

const char* upload_kind_to_str(UploadKind k) {
    switch (k) {
        case UploadKind::ITEM_IN:    return "ITEM_IN";
        case UploadKind::ITEM_OUT:   return "ITEM_OUT";
        case UploadKind::ITEM_MOVED: return "ITEM_MOVED";
        case UploadKind::DOOR_CLOSE: return "DOOR_CLOSE";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
//  base64 编码（标准表，带 '=' 补齐）
// ---------------------------------------------------------------------------
static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const unsigned char* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= len) {
        unsigned int n = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        out.push_back(B64[(n >> 18) & 0x3F]);
        out.push_back(B64[(n >> 12) & 0x3F]);
        out.push_back(B64[(n >> 6) & 0x3F]);
        out.push_back(B64[n & 0x3F]);
        i += 3;
    }
    size_t rem = len - i;
    if (rem == 1) {
        unsigned int n = data[i] << 16;
        out.push_back(B64[(n >> 18) & 0x3F]);
        out.push_back(B64[(n >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    } else if (rem == 2) {
        unsigned int n = (data[i] << 16) | (data[i + 1] << 8);
        out.push_back(B64[(n >> 18) & 0x3F]);
        out.push_back(B64[(n >> 12) & 0x3F]);
        out.push_back(B64[(n >> 6) & 0x3F]);
        out.push_back('=');
    }
    return out;
}

// ---------------------------------------------------------------------------
CloudUploader::CloudUploader()
    : running_(false), sent_ok_(0), sent_fail_(0) {
    // 默认配置来自 fridge_config.h；环境变量可覆盖（方便现场改 IP 不用重编译）
    host       = CLOUD_HOST;
    port       = CLOUD_PORT;
    item_path  = CLOUD_ITEM_PATH;
    inventory_path = CLOUD_INVENTORY_PATH;
    device_id  = CLOUD_DEVICE_ID;

    if (const char* h = getenv("FRIDGE_CLOUD_HOST"))   host = h;
    if (const char* p = getenv("FRIDGE_CLOUD_PORT"))   port = atoi(p);
    if (const char* ip = getenv("FRIDGE_ITEM_PATH"))   item_path = ip;
    if (const char* ip = getenv("FRIDGE_INVENTORY_PATH")) inventory_path = ip;
    if (const char* d = getenv("FRIDGE_DEVICE_ID"))    device_id = d;
}

CloudUploader::~CloudUploader() { stop(); }

void CloudUploader::start() {
    if (running_.load()) return;
    running_.store(true);
    thread_ = std::thread(&CloudUploader::worker_loop, this);
    printf("[CLOUD] uploader started → http://%s:%d%s, inventory=%s (device=%s)\n",
           host.c_str(), port, item_path.c_str(), inventory_path.c_str(),
           device_id.c_str());
}

void CloudUploader::stop() {
    if (!running_.load()) return;
    running_.store(false);
    in_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void CloudUploader::enqueue(const UploadJob& job) {
    {
        std::lock_guard<std::mutex> lk(in_mtx_);
        while (jobs_.size() >= MAX_QUEUE) jobs_.pop_front();  // 满了丢最旧
        jobs_.push_back(job);
    }
    in_cv_.notify_one();
}

void CloudUploader::enqueue_inventory_snapshot(const std::string& json,
                                               long long timestamp_ms) {
    UploadJob job;
    job.kind = UploadKind::DOOR_CLOSE;
    job.timestamp_ms = timestamp_ms;
    job.raw_json = json;
    job.max_attempts = 3;
    enqueue(job);
}

// ---------------------------------------------------------------------------
//  组装 JSON —— 严格对应《端侧返回数据格式.txt》出入库事件：
//    {"device_id","timestamp","event_type","data":[
//       {"local_track_id","category","confidence","bbox":[x,y,w,h],"crop_image"}]}
//  crop_image 仅 ITEM_IN 带（jpeg 非空时）；ITEM_OUT/MOVED 为空字符串。
// ---------------------------------------------------------------------------
std::string CloudUploader::build_json(const UploadJob& job) {
    if (job.kind == UploadKind::DOOR_CLOSE && !job.raw_json.empty()) {
        return job.raw_json;
    }

    std::string b64;
    if (!job.jpeg.empty()) b64 = base64_encode(job.jpeg.data(), job.jpeg.size());

    std::string s;
    s.reserve(b64.size() + 320);
    char head[640];
    snprintf(head, sizeof(head),
             "{\"device_id\":\"%s\",\"timestamp\":%lld,"
             "\"event_type\":\"%s\",\"data\":[{"
             "\"local_track_id\":%d,\"category\":\"%s\","
             "\"confidence\":%.2f,\"bbox\":[%d,%d,%d,%d],\"crop_image\":\"",
             device_id.c_str(), job.timestamp_ms,
             upload_kind_to_str(job.kind),
             job.local_track_id, job.category.c_str(), job.confidence,
             job.x, job.y, job.w, job.h);
    s += head;
    s += b64;   // 可能为空字符串
    s += "\"}]}";
    return s;
}

bool CloudUploader::http_post(const std::string& path, const std::string& body,
                              std::string& resp_body) {
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
    char header[512];
    snprintf(header, sizeof(header),
             "POST %s HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n"
             "\r\n",
             path.c_str(), host.c_str(), port, body.size());

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

void CloudUploader::worker_loop() {
    while (running_.load()) {
        UploadJob job;
        {
            std::unique_lock<std::mutex> lk(in_mtx_);
            in_cv_.wait(lk, [this] { return !jobs_.empty() || !running_.load(); });
            if (!running_.load() && jobs_.empty()) break;
            if (jobs_.empty()) continue;
            job = jobs_.front();
            jobs_.pop_front();
        }

        std::string body = build_json(job);
        std::string resp;
        const std::string& path =
            job.kind == UploadKind::DOOR_CLOSE ? inventory_path : item_path;
        bool ok = http_post(path, body, resp);

        if (ok) {
            sent_ok_++;
            if (job.kind == UploadKind::DOOR_CLOSE) {
                printf("[CLOUD] 关门库存快照上报成功 (%zuB json)\n", body.size());
            } else {
                printf("[CLOUD] 事件 %s (item#%d, %zuB jpeg) 上报成功\n",
                       upload_kind_to_str(job.kind), job.local_track_id,
                       job.jpeg.size());
            }
        } else {
            if (job.attempts + 1 < job.max_attempts) {
                job.attempts++;
                printf("[CLOUD] %s 上报失败，准备第 %d/%d 次重试\n",
                       upload_kind_to_str(job.kind), job.attempts + 1,
                       job.max_attempts);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                {
                    std::lock_guard<std::mutex> lk(in_mtx_);
                    while (jobs_.size() >= MAX_QUEUE) jobs_.pop_front();
                    jobs_.push_back(job);
                }
                in_cv_.notify_one();
            } else {
                sent_fail_++;
                printf("[CLOUD] %s 上报失败（已达到最大重试次数）\n",
                       upload_kind_to_str(job.kind));
            }
        }
    }
    printf("[CLOUD] uploader stopped (ok=%d, fail=%d)\n",
           sent_ok_.load(), sent_fail_.load());
}

}  // namespace fridge
