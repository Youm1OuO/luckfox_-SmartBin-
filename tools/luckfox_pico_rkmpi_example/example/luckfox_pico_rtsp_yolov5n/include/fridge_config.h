// ============================================================================
//  fridge_config.h
//  冰箱视觉系统 - 全局配置
//  在这一个文件里集中所有"魔术数字"和类别 ID 配置，方便后续调参。
// ============================================================================
#ifndef __FRIDGE_CONFIG_H
#define __FRIDGE_CONFIG_H

namespace fridge {

// =========================================================================
//  类别 ID 配置
// -------------------------------------------------------------------------
//  说明：YOLO 模型输出的 cls_id 是一个整数，对应 coco_80_labels_list.txt
//  里的行号(0-indexed)。我们需要明确"哪个 cls_id 算手、哪些算食材"。
//
//  当前演示阶段：直接复用 COCO 80 类模型，把 person(cls_id=0) 当作 hand。
//  后续训练自己的冰箱模型时，把 hand 训为第 0 类，其余类别为各种食材。
// =========================================================================
constexpr int CLASS_HAND = 33;          // 手所对应的 cls_id

// 工具函数：判断一个 cls_id 是不是"手"
inline bool is_hand(int cls_id) {
    return cls_id == CLASS_HAND;
}

// 工具函数：判断一个 cls_id 是不是"食材/物品"
// 简单粗暴：只要不是手、并且 cls_id 合法，就当作食材
inline bool is_food(int cls_id) {
    return cls_id >= 0 && cls_id != CLASS_HAND;
}

// =========================================================================
//  YOLO 输出阈值
// -------------------------------------------------------------------------
//  分三层：
//    1. YOLO_CANDIDATE_SCORE_THRESH：RKNN 原始输出的粗筛，低一点，避免过早丢候选。
//    2. YOLO_OBJECT_SCORE_THRESH / YOLO_HAND_SCORE_THRESH：postprocess 最终输出阈值。
//    3. 业务层手阈值：HAND_CONTEXT_SCORE_THRESH；检测到就停止稳定快照。
// =========================================================================
constexpr float YOLO_CANDIDATE_SCORE_THRESH = 0.01f;
constexpr float YOLO_OBJECT_SCORE_THRESH    = 0.25f;    // 物品的阈值
constexpr float YOLO_HAND_SCORE_THRESH      = 0.30f;    // 手的阈值

inline float yolo_output_score_threshold(int cls_id) {
    return is_hand(cls_id) ? YOLO_HAND_SCORE_THRESH : YOLO_OBJECT_SCORE_THRESH;
}

// =========================================================================
//  粗粒度分类映射（对应 classes.yaml 的 coarse_grained）
// -------------------------------------------------------------------------
//  赛题要求至少输出粗类（蔬果 / 肉蛋生鲜 / 饮料乳品 / 包装食品）。
//  这里把 43 个细类 ID 映射到 5 个粗类字符串，库存上报/展示用。
//  细类 ID 必须与 model/labels_list.txt 的行号严格对应。
// =========================================================================
inline const char* coarse_category(int cls_id) {
    switch (cls_id) {
        // 蔬果类: apple..leafy_green(0..17) + mushroom..okra(34..42)
        case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
        case 8: case 9: case 10: case 11: case 12: case 13: case 14:
        case 15: case 16: case 17:
        case 34: case 35: case 36: case 37: case 38: case 39: case 40:
        case 41: case 42:
        case 43: case 44: case 45: case 46: case 47:
            return "fruit_veg";
        // 肉蛋生鲜类: egg, meat_pack, fish_pack(18..20)
        case 18: case 19: case 20:
            return "meat_seafood";
        // 饮料乳品类: milk_box..water_bottle(21..26)
        case 21: case 22: case 23: case 24: case 25: case 26:
            return "beverage_dairy";
        // 包装食品类: bagged_food..fresh_box(27..32)
        case 27: case 28: case 29: case 30: case 31: case 32:
            return "packaged_food";
        // 手
        case CLASS_HAND:
            return "interference";
        default:
            return "unknown";
    }
}

// =========================================================================
//  中文类别映射 — 对应后端接口要求的 category 字段
// -------------------------------------------------------------------------
//  后端要求中文细粒度（如"苹果"），端侧 labels_list.txt 是英文。
//  此函数将 cls_id 映射为中文类别名，用于上报给后端。
// =========================================================================
inline const char* cls_id_to_chinese(int cls_id) {
    static const char* cn_names[] = {
        "苹果", "香蕉", "橙子", "番茄", "柠檬",
        "梨", "葡萄", "草莓", "西瓜", "哈密瓜",
        "木瓜", "牛油果", "黄瓜", "胡萝卜", "土豆",
        "洋葱", "甜椒", "叶菜", "鸡蛋", "包装肉",
        "包装鱼", "盒装牛奶", "瓶装牛奶", "酸奶杯",
        "果汁瓶", "苏打罐", "水瓶", "袋装食品",
        "盒装食品", "罐头", "玻璃罐", "保鲜膜",
        "保鲜盒", "手", "蘑菇", "南瓜", "大蒜",
        "姜", "萝卜", "红薯", "核桃", "香菜",
        "秋葵", "白菜", "豆角", "生菜", "卷心菜", "苦瓜"
    };
    if (cls_id >= 0 && cls_id < 48) return cn_names[cls_id];
    return "未知";
}

// =========================================================================
//  标签扫描 — 哪些类别"可能有包装标签值得扫"
// -------------------------------------------------------------------------
//  "内容不易直接辨识"的包装类食品(袋装/盒装/罐装/瓶装/保鲜盒)。
//  当前业务路径不依赖它（所有放入物品都带截图给后端，由后端决定怎么处理），
//  保留此函数供以后扩展使用（例如 UI 上标注"此物品可能有标签"）。
// =========================================================================
inline bool has_label(int cls_id) {
    switch (cls_id) {
        case 19: // meat_pack 包装肉
        case 20: // fish_pack 包装鱼
        case 21: // milk_box 盒装牛奶
        case 24: // juice_bottle 果汁瓶
        case 27: // bagged_food 袋装食品
        case 28: // boxed_food 盒装食品
        case 29: // canned_food 罐头
        case 30: // jar_food 玻璃罐
        case 31: // plastic_wrap 保鲜膜包裹
        case 32: // fresh_box 保鲜盒
            return true;
        default:
            return false;
    }
}

// =========================================================================
//  端云协同 — 云端服务默认配置（可被环境变量覆盖，见 cloud_uploader.cc）
// -------------------------------------------------------------------------
//  现场改 IP 不想重新编译时，直接设环境变量：
//    export FRIDGE_CLOUD_HOST=192.168.5.6
//    export FRIDGE_CLOUD_PORT=8000
//  上报端点：
//    POST /api/v1/admin/device-ingest      单个事件 ITEM_IN/OUT/MOVED
//    POST /api/v1/admin/events/heartbeat    设备心跳（每30秒）
//  鉴权：所有接口均需 Authorization: Bearer <token> 请求头
// =========================================================================
constexpr const char* CLOUD_HOST       = "192.168.168.1";     // 板子通过 USB 网络访问宿主机，由宿主机转发到后端
constexpr int         CLOUD_PORT       = 8000;                // 端口
constexpr const char* CLOUD_ITEM_PATH  = "/api/v1/admin/device-ingest"; // 出入库事件端点
constexpr const char* CLOUD_HEARTBEAT_PATH = "/api/v1/admin/events/heartbeat"; // 心跳端点
constexpr const char* CLOUD_LOGIN_PATH = "/api/v1/admin/auth/login"; // 登录端点
constexpr const char* CLOUD_INVENTORY_PATH = "/inventory/close"; // 关门库存端点
constexpr const char* CLOUD_DEVICE_ID  = "luckfox-001";       // 设备 ID
constexpr const char* CLOUD_USERNAME   = "admin";              // 登录用户名
constexpr const char* CLOUD_PASSWORD   = "admin123";           // 登录密码
constexpr int         HEARTBEAT_INTERVAL_SEC = 30;             // 心跳间隔（秒）

// =========================================================================
//  推理输入颜色顺序开关（已实测定论）
// -------------------------------------------------------------------------
//  结论：保持 BGR 直接喂模型是正确的。实测把输入转成 RGB(=true) 后，
//  连苹果都识别不到，置信度全面下降 → 说明该 rknn 模型期望的就是 BGR 输入，
//  原链路(frame 为 BGR 直接喂)没有红蓝反的问题。
//    false = 保持 BGR 直接喂（正确，当前用这个）
//    true  = 推理前 BGR→RGB（实测会让识别变差，勿用）
//  保留此开关仅作记录，避免日后再次怀疑通道顺序。
// =========================================================================
constexpr bool INFER_INPUT_BGR2RGB = false;

// =========================================================================
//  历史 ByteTrack-Lite 参数（当前 yolov5n 目标不会编译 tracker.cc）
// =========================================================================
// 高分检测阈值。score >= HIGH_SCORE_THRESH 的 detection 进入第一轮匹配，
// score < HIGH_SCORE_THRESH 但 >= LOW_SCORE_THRESH 的 detection 进入第二轮。
constexpr float HIGH_SCORE_THRESH = 0.5f;
constexpr float LOW_SCORE_THRESH  = 0.1f;

// 第一轮匹配的最低 IoU（IoU 低于此值的 pair 不允许配对）
constexpr float MATCH_IOU_THRESH       = 0.30f;
// 第二轮匹配（低分检测 vs 未匹配 track）的最低 IoU，可以更宽松
constexpr float MATCH_IOU_THRESH_LOW   = 0.20f;
// 第三轮：未匹配的高分检测 vs 未确认 track（unconfirmed）
constexpr float MATCH_IOU_THRESH_UNCFM = 0.50f;

// Track 丢失多少帧后销毁。10 FPS 下 60 帧 ≈ 6 秒。
// 冰箱场景里手会遮挡较久，给宽一点。
constexpr int TRACK_BUFFER_FRAMES = 60;

// 新 detection 升级为新 track 的最低分数（避免低分检测立刻成 track）
constexpr float NEW_TRACK_SCORE_THRESH = 0.6f;

// =========================================================================
//  稳态切片 + 会话事件
// =========================================================================

// =========================================================================
//  连续无手稳定快照
// =========================================================================
constexpr int   SNAPSHOT_N            = 4;        // 连续无手稳定帧数；不要求是奇数
constexpr float SNAPSHOT_S            = 0.6f;     // 投票阈值百分比（N=4 时至少出现 3 帧）
constexpr float SNAPSHOT_MIN_SCORE    = 0.5f;     // 最低检测分数（低于此分数的不进快照）
constexpr long long FIRST_SNAPSHOT_EMPTY_GRACE_MS = 800LL; // 开门曝光稳定前不急着用0件快照判空
// 仅用于快照聚类：把完整框与被包含的局部框归到同一候选。
constexpr float PARTIAL_MATCH_CONTAINMENT    = 0.85f;
constexpr float PARTIAL_MATCH_MAX_AREA_RATIO = 0.90f;

// =========================================================================
//  手框
// =========================================================================
// 手证据阈值：业务层只要检测到这种手框，就停止生成稳定快照。
constexpr float HAND_CONTEXT_SCORE_THRESH = YOLO_HAND_SCORE_THRESH;
// 强手框仅用于 OSD 显示，避免 UI 被普通小手框干扰。
constexpr float OSD_STRONG_HAND_SCORE_THRESH = 0.45f;
constexpr float OSD_STRONG_HAND_MIN_AREA_RATIO = 0.002f;

// =========================================================================
//  稳定快照与库存的单框关系
// -------------------------------------------------------------------------
//  NORMAL / SHRINK / GROW 的所有阈值集中在这里；数值只作为初始值，
//  必须用真实视频继续标定。
// =========================================================================
constexpr float SNAPSHOT_NORMAL_IOM          = 0.70f;
constexpr float SNAPSHOT_CONTAIN_IOM         = 0.80f;
constexpr float SNAPSHOT_CENTER_NORMAL        = 0.18f;
constexpr float SNAPSHOT_CENTER_CONTAIN       = 0.55f;
constexpr float SNAPSHOT_SHAPE_NORMAL         = 0.28f;
constexpr float SNAPSHOT_SHAPE_CONTAIN        = 0.85f;
// NORMAL 与 SHRINK / GROW 留出安全间隔，避免一个框同时落入两种关系。
constexpr float SNAPSHOT_NORMAL_AREA_MIN      = 0.91f;
constexpr float SNAPSHOT_NORMAL_AREA_MAX      = 1.09f;
constexpr float SNAPSHOT_SHRINK_AREA_MAX      = 0.88f;
constexpr float SNAPSHOT_GROW_AREA_MIN        = 1.12f;
constexpr float SNAPSHOT_GROW_AREA_MAX        = 4.00f;
constexpr float SNAPSHOT_BLOCK_COVER          = 0.55f;
constexpr float SNAPSHOT_FULL_COVER           = 0.80f;
constexpr float SNAPSHOT_LEAVE_COVER_MAX      = 0.15f;

// =========================================================================
//  OperationTrack 参数
// -------------------------------------------------------------------------
// Track 只辅助确认 MOVED；它不参与快照中的 NORMAL / SHRINK / GROW 裁决。
// =========================================================================
constexpr float TRACK_REAPPEAR_CENTER_NORM  = 1.35f;
constexpr float TRACK_FRAME_CENTER_NORM     = 0.85f;
constexpr float TRACK_FRAME_WIDTH_RATIO     = 0.85f;
constexpr float TRACK_FRAME_HEIGHT_RATIO    = 0.85f;
constexpr float TRACK_HAND_NEAR_NORM        = 1.40f;
constexpr float TRACK_HAND_OVERLAP           = 0.12f;
// Candidate 升级为正式 Track 所需的最小位移；比“静止”阈值宽，避免 YOLO 微抖动建轨。
constexpr float TRACK_CREATE_MOTION_NORM     = 0.25f;
// 完全遮挡 Candidate 必须被手覆盖较大部分，不能只因手在附近就建立。
constexpr float TRACK_FULL_OCCLUSION_OVERLAP = 0.50f;
constexpr float TRACK_PLACED_SIZE_RATIO     = 0.50f;

// =========================================================================
//  帧尺寸（用于手部扩展等计算）
// =========================================================================
constexpr float FRAME_W = 1280.0f;
constexpr float FRAME_H = 720.0f;

}  // namespace fridge

#endif  // __FRIDGE_CONFIG_H
