// ============================================================================
//  cloud_uploader.h
//  端云协同 — 异步出入库事件上报器
//
//  设计目标（严格对应后端接口文档）：
//    - 端侧只做"检测 + 裁剪 + 上传"，不在板子上跑 OCR（RV1106 跑不动）。
//      放入(ITEM_IN) 带物品框内截图(crop_image)，后端拿到图后自行决定要不要
//      跑云端 AI 读标签 / 重识别 / 存档；端侧不参与标签识别与配对。
//    - 绝不阻塞 NPU 推理主链路：上传走独立后台线程 + 任务队列，推理线程只管
//      把"裁剪好的 JPEG + 元数据"丢进队列，立即返回。
//    - 上报端点：POST /api/v1/admin/device-ingest
//    - 鉴权：Authorization: Bearer <token>
//    - 心跳：POST /api/v1/admin/events/heartbeat（每30秒）
// ============================================================================
#ifndef __FRIDGE_CLOUD_UPLOADER_H
#define __FRIDGE_CLOUD_UPLOADER_H

#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace fridge {

// 上传任务类型 —— 对应后端接口的 event_type
enum class UploadKind {
    ITEM_IN,       // 放入（带截图）
    ITEM_OUT,      // 取出（不带图）
    ITEM_MOVED,    // 挪位（带整理前后框 + 整理后截图）
    NO_EVENT_SNAPSHOT, // 无事件稳定时的整柜截图
    DOOR_CLOSE,    // 关门时最终库存快照
};

// 一个待上传任务（端 → 云）
struct UploadJob {
    UploadKind   kind = UploadKind::ITEM_IN;
    int          local_track_id = -1;   // 端侧追踪临时 ID（防抖去重，= 库存 item_id）
    std::string  category;              // 中文细粒度分类（如"苹果"）
    float        confidence = 0.0f;     // YOLO 识别置信度 [0,1]
    int          x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // bbox [x1, y1, x2, y2]（原图像素）
    bool         has_before_bbox = false;
    int          before_x1 = 0, before_y1 = 0, before_x2 = 0, before_y2 = 0;
    float        pixel_diff = 0.0f;     // NO_EVENT_SNAPSHOT 使用：稳定画面平均像素差
    std::string  snapshot_reason;       // NO_EVENT_SNAPSHOT 使用
    long long    timestamp_ms = 0;      // 事件发生时间戳（毫秒）
    std::vector<unsigned char> jpeg;    // 事件相关 JPEG 图
    std::string  raw_json;              // DOOR_CLOSE 使用：完整库存 JSON
    int          attempts = 0;           // 后台线程重试计数
    int          max_attempts = 3;       // 网络临时失败时最多尝试次数
};

// 端云协同上传器：后台线程异步 POST，不阻塞推理主链路
class CloudUploader {
public:
    CloudUploader();
    ~CloudUploader();

    // 启动 / 停止后台线程
    void start();
    void stop();

    // 登录获取 Token（启动时调用一次）
    bool login();

    // 推理线程调用：把一个上传任务塞进队列（深拷贝，立即返回）
    // 队列满时丢弃最旧任务，保证不会无限堆积撑爆内存
    void enqueue(const UploadJob& job);

    // 关门线程调用：上传最终库存快照
    void enqueue_inventory_snapshot(const std::string& json, long long timestamp_ms);

    // ---- 配置（构造后、start 前可改；默认值见 fridge_config.h）----
    // 也可被环境变量覆盖：FRIDGE_CLOUD_HOST / FRIDGE_CLOUD_PORT /
    //                      FRIDGE_ITEM_PATH / FRIDGE_DEVICE_ID
    std::string host;
    int         port = 8000;
    std::string item_path;        // 出入库事件端点
    std::string heartbeat_path;   // 心跳端点
    std::string login_path;       // 登录端点
    std::string inventory_path;   // 关门库存端点
    std::string device_id;
    std::string username;         // 登录用户名
    std::string password;         // 登录密码
    std::string token;            // 登录后获取的 token

    // 统计（调试用）
    int sent_ok()   const { return sent_ok_.load(); }
    int sent_fail() const { return sent_fail_.load(); }

private:
    void worker_loop();
    void heartbeat_loop();
    // 发送一条 HTTP/1.1 POST（application/json）到指定 path，返回是否成功 + 响应体
    bool http_post(const std::string& path, const std::string& body,
                   std::string& resp_body, bool use_auth = true);
    // 把一个 UploadJob 组装成符合后端接口的 JSON
    std::string build_json(const UploadJob& job);

    std::thread             thread_;
    std::thread             heartbeat_thread_;
    std::atomic<bool>       running_;
    std::atomic<bool>       logged_in_;

    std::mutex              in_mtx_;
    std::condition_variable in_cv_;
    std::deque<UploadJob>   jobs_;
    static constexpr size_t MAX_QUEUE = 16;   // 队列上限，超了丢最旧

    std::atomic<int> sent_ok_;
    std::atomic<int> sent_fail_;
};

// event_type 字符串（导出供日志/测试复用）
const char* upload_kind_to_str(UploadKind k);

// base64 编码工具（导出供单元测试或其他模块复用）
std::string base64_encode(const unsigned char* data, size_t len);

}  // namespace fridge

#endif  // __FRIDGE_CLOUD_UPLOADER_H
