// ============================================================================
//  cloud_json.cc
//  Cloud upload JSON, Base64, and response-field helpers
// ============================================================================
#include "cloud_uploader.h"

#include <cstdio>

namespace fridge {

const char* upload_kind_to_str(UploadKind k) {
    switch (k) {
        case UploadKind::ITEM_IN:    return "ITEM_IN";
        case UploadKind::ITEM_OUT:   return "ITEM_OUT";
        case UploadKind::ITEM_MOVED: return "ITEM_MOVED";
        case UploadKind::NO_EVENT_SNAPSHOT: return "NO_EVENT_SNAPSHOT";
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
//  简易 JSON 解析（只提取顶层 string/int 字段，不引第三方库）
// ---------------------------------------------------------------------------
std::string json_extract_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

// ---------------------------------------------------------------------------
//  组装 JSON —— 严格对应后端接口文档：
//    {"device_id","timestamp","event_type","data":[
//       {"local_track_id","category","confidence","bbox":[x1,y1,x2,y2],"crop_image"}]}
//  ITEM_MOVED 额外带 before_bbox / before_position / after_bbox /
//  after_position / after_image。
// ---------------------------------------------------------------------------
std::string CloudUploader::build_json(const UploadJob& job) {
    if (job.kind == UploadKind::DOOR_CLOSE && !job.raw_json.empty()) {
        return job.raw_json;
    }

    std::string b64;
    if (!job.jpeg.empty()) b64 = base64_encode(job.jpeg.data(), job.jpeg.size());

    if (job.kind == UploadKind::NO_EVENT_SNAPSHOT) {
        std::string reason = job.snapshot_reason.empty()
            ? "stable_no_event" : job.snapshot_reason;
        std::string s;
        s.reserve(b64.size() + 320);
        char head[512];
        snprintf(head, sizeof(head),
                 "{\"device_id\":\"%s\",\"timestamp\":%lld,"
                 "\"event_type\":\"%s\",\"data\":[{"
                 "\"reason\":\"%s\",\"pixel_diff\":%.2f,\"image\":\"",
                 device_id.c_str(), job.timestamp_ms,
                 upload_kind_to_str(job.kind), reason.c_str(), job.pixel_diff);
        s += head;
        s += b64;
        s += "\"}]}";
        return s;
    }

    std::string s;
    s.reserve(b64.size() + 480);
    char head[640];
    snprintf(head, sizeof(head),
             "{\"device_id\":\"%s\",\"timestamp\":%lld,"
             "\"event_type\":\"%s\",\"data\":[{"
             "\"local_track_id\":%d,\"category\":\"%s\","
             "\"confidence\":%.2f,\"bbox\":[%d,%d,%d,%d]",
             device_id.c_str(), job.timestamp_ms,
             upload_kind_to_str(job.kind),
             job.local_track_id, job.category.c_str(), job.confidence,
             job.x1, job.y1, job.x2, job.y2);
    s += head;

    if (job.kind == UploadKind::ITEM_MOVED && job.has_before_bbox) {
        int before_cx = (job.before_x1 + job.before_x2) / 2;
        int before_cy = (job.before_y1 + job.before_y2) / 2;
        int after_cx = (job.x1 + job.x2) / 2;
        int after_cy = (job.y1 + job.y2) / 2;
        char moved_fields[384];
        snprintf(moved_fields, sizeof(moved_fields),
                 ",\"before_bbox\":[%d,%d,%d,%d],"
                 "\"before_position\":[%d,%d],"
                 "\"after_bbox\":[%d,%d,%d,%d],"
                 "\"after_position\":[%d,%d]",
                 job.before_x1, job.before_y1, job.before_x2, job.before_y2,
                 before_cx, before_cy,
                 job.x1, job.y1, job.x2, job.y2,
                 after_cx, after_cy);
        s += moved_fields;
    }

    if (job.kind == UploadKind::ITEM_MOVED) {
        s += ",\"after_image\":\"";
        s += b64;
        s += "\"";
    } else {
        s += ",\"crop_image\":\"";
        s += b64;   // ITEM_IN 有图；ITEM_OUT 为空字符串
        s += "\"";
    }

    s += "}]}";
    return s;
}

}  // namespace fridge
