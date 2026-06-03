// ============================================================================
//  cloud_uploader.h
//  端云协同 — 异步出入库事件上报器
//
//  设计目标（严格对应《端侧返回数据格式.txt》）：
//    - 端侧只做"检测 + 裁剪 + 上传"，不在板子上跑 OCR（RV1106 跑不动）。
//      放入(ITEM_IN) 带物品框内截图(crop_image)，后端拿到图后自行决定要不要
//      跑云端 AI 读标签 / 重识别 / 存档；端侧不参与标签识别与配对。
//    - 绝不阻塞 NPU 推理主链路：上传走独立后台线程 + 任务队列，推理线程只管
//      把"裁剪好的 JPEG + 元数据"丢进队列，立即返回。
//    - 上报端点：POST /events/item，event_type = ITEM_IN / ITEM_OUT / ITEM_MOVED
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

// 上传任务类型 —— 对应《端侧返回数据格式.txt》的 event_type
enum class UploadKind {
    ITEM_IN,       // 放入（带截图）
    ITEM_OUT,      // 取出（不带图）
    ITEM_MOVED,    // 挪位（不带图）
};

// 一个待上传任务（端 → 云）
struct UploadJob {
    UploadKind   kind = UploadKind::ITEM_IN;
    int          local_track_id = -1;   // 端侧追踪临时 ID（防抖去重，= 库存 item_id）
    std::string  category;              // 粗粒度分类（fruit_veg/meat_seafood/...）
    float        confidence = 0.0f;     // YOLO 识别置信度 [0,1]
    int          x = 0, y = 0, w = 0, h = 0;  // bbox [x, y, w, h]（原图像素）
    long long    timestamp_ms = 0;      // 事件发生时间戳（毫秒）
    std::vector<unsigned char> jpeg;    // 物品框局部截图 JPEG（仅 ITEM_IN 带；可空）
};

// 端云协同上传器：后台线程异步 POST，不阻塞推理主链路
class CloudUploader {
public:
    CloudUploader();
    ~CloudUploader();

    // 启动 / 停止后台线程
    void start();
    void stop();

    // 推理线程调用：把一个上传任务塞进队列（深拷贝，立即返回）
    // 队列满时丢弃最旧任务，保证不会无限堆积撑爆内存
    void enqueue(const UploadJob& job);

    // ---- 配置（构造后、start 前可改；默认值见 fridge_config.h）----
    // 也可被环境变量覆盖：FRIDGE_CLOUD_HOST / FRIDGE_CLOUD_PORT /
    //                      FRIDGE_ITEM_PATH / FRIDGE_DEVICE_ID
    std::string host;
    int         port = 8000;
    std::string item_path;       // 出入库事件端点，默认 /events/item
    std::string device_id;

    // 统计（调试用）
    int sent_ok()   const { return sent_ok_.load(); }
    int sent_fail() const { return sent_fail_.load(); }

private:
    void worker_loop();
    // 发送一条 HTTP/1.1 POST（application/json）到指定 path，返回是否成功 + 响应体
    bool http_post(const std::string& path, const std::string& body,
                   std::string& resp_body);
    // 把一个 UploadJob 组装成符合《端侧返回数据格式.txt》的 JSON
    std::string build_json(const UploadJob& job);

    std::thread             thread_;
    std::atomic<bool>       running_;

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
