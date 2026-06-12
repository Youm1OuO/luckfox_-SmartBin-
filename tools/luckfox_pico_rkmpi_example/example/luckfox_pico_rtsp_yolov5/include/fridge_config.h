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
//    3. 业务层手阈值：HAND_SNAPSHOT_BLOCK_*，只用于决定是否阻塞快照。
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
//    export FRIDGE_CLOUD_HOST=192.168.168.1
//    export FRIDGE_CLOUD_PORT=8000
//  上报端点：
//    POST /events/item      单个事件 ITEM_IN/OUT/MOVED
//    POST /inventory/close  关门时最终库存快照
//  放入(ITEM_IN) 带物品框内截图，后端拿到图后自行决定要不要跑云端 AI。
// =========================================================================
constexpr const char* CLOUD_HOST       = "192.168.168.1";   // 云端/网关主机
constexpr int         CLOUD_PORT       = 8000;              // 端口
constexpr const char* CLOUD_ITEM_PATH  = "/events/item";    // 出入库事件端点
constexpr const char* CLOUD_INVENTORY_PATH = "/inventory/close"; // 关门库存端点
constexpr const char* CLOUD_DEVICE_ID  = "luckfox";         // 设备 ID

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
//  ByteTrack-Lite 跟踪器参数
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
//  新业务流程7：严格原位身份匹配阈值
// -------------------------------------------------------------------------
//  判断"这个新检测到的东西，是不是库存里那个旧东西"。
//  原位匹配只用于判断"位置基本没变"：
//    1. 类别相同
//    2. 中心距离 < IDENTITY_CENTER_DIST
//    3. 面积比差异 < IDENTITY_AREA_RATIO
//    4. IoU > IDENTITY_IOU_THRESH
//  注意：整理/移动不允许只靠这里的外观相似直接确认。
// =========================================================================
constexpr float IDENTITY_CENTER_DIST  = 15.0f;   // ε1: 中心距离阈值（像素）
constexpr float IDENTITY_AREA_RATIO   = 0.2f;    // ε2: 面积比差异阈值 (|A-B|/max(A,B))
constexpr float IDENTITY_IOU_THRESH   = 0.5f;    // ε3: IoU 阈值（从0.7降到0.5，容忍bbox微变）
constexpr float IDENTITY_COLOR_DIFF   = 20.0f;   // 像素颜色差异阈值（每通道 0~255）

// =========================================================================
//  新业务流程7：整理 / 移动判定阈值
// -------------------------------------------------------------------------
//  reid_match 只是外观输入，confirmed_relocation 还必须有移动/手/HELD
//  等 OperationContext 证据，并且候选唯一性足够好。
// =========================================================================
constexpr float RELOCATION_REID_MIN          = 0.70f;
constexpr float RELOCATION_REID_STRONG       = 0.82f;
constexpr float RELOCATION_REID_MARGIN       = 0.10f;
constexpr float RELOCATION_EVIDENCE_WEAK     = 0.25f;
constexpr float RELOCATION_EVIDENCE_STRONG   = 0.75f;
constexpr int   PENDING_RELOCATION_CONFIRM_FRAMES = 1;
constexpr int   PENDING_RELOCATION_EXPIRE_FRAMES  = 2;
constexpr int   PENDING_NEW_CONFIRM_FRAMES = 1;
constexpr int   PENDING_NEW_EXPIRE_FRAMES  = 3;

// =========================================================================
//  新业务流程7：多帧快照投票
// =========================================================================
constexpr int   SNAPSHOT_N            = 3;        // N帧为一个快照（必须为奇数）
constexpr float SNAPSHOT_OBJECT_STABLE_RATIO = 0.6f; // 物体出现稳定阈值（5帧中至少3帧）
constexpr float SNAPSHOT_MIN_SCORE    = 0.3f;     // 最低检测分数（低于此分数的不进快照）
constexpr int   SNAPSHOT_HAND_BLOCK_MIN_COUNT = 1; // N帧中至少几帧有阻塞手，快照才算带手
constexpr long long FIRST_SNAPSHOT_EMPTY_GRACE_MS = 800LL; // 开门曝光稳定前不急着用0件快照判空

// 跨类别 spatial cluster：只处理同一位置、同一大小、同一框形状的类别抖动。
// 注意：这些阈值和 SNAPSHOT_OBJECT_STABLE_RATIO 解耦。
//   - OBJECT_STABLE 判断“这个物体是否稳定出现”
//   - CLASS_STABLE 判断“同一个 spatial cluster 的类别投票是否稳定”
// 即便两个阈值当前数值相同，也不要共用一个变量。
constexpr float SNAPSHOT_CLASS_CONFLICT_IOU = 0.75f;          // 跨类别归并要求的最小 IoU
constexpr float SNAPSHOT_CLASS_CONFLICT_CENTER_RATIO = 0.25f; // 中心距离 / 小框对角线
constexpr float SNAPSHOT_CLASS_CONFLICT_AREA_RATIO = 0.35f;   // 面积差异比例
constexpr float SNAPSHOT_CLASS_CONFLICT_MOTION_RATIO = 0.35f; // 整个cluster移动过大则不做类别抖动归并
constexpr float SNAPSHOT_CLASS_STABLE_RATIO = 0.6f;           // 最终类别票数 / cluster总票数
constexpr int   SNAPSHOT_CLASS_COUNT_MARGIN = 1;              // 第一名类别至少比第二名多几票

// =========================================================================
//  新业务流程7：自适应"附近"距离阈值
// -------------------------------------------------------------------------
//  normalized_nearby_distance = center_distance / min(对角线A, 对角线B)
//  小于此值判定为"附近"
// =========================================================================
constexpr float NEARBY_DISTANCE_THRESH = 1.0f;  // 设计文档建议值，可微调

// =========================================================================
//  新业务流程7：OperationContext / HELD 证据相关
// =========================================================================
// 连续多少帧没有检测到手 → 确认手离开画面
constexpr int HAND_LEAVE_FRAMES  = 5;
// 手证据阈值：用于 OperationContext / OSD，和 YOLO_HAND_SCORE_THRESH 保持一致即可。
constexpr float HAND_CONTEXT_SCORE_THRESH = YOLO_HAND_SCORE_THRESH;
// 阻塞快照的手要更严格，避免低置信/小框误检长期卡住库存对比。
constexpr float HAND_SNAPSHOT_BLOCK_SCORE_THRESH = 0.45f;
constexpr float HAND_SNAPSHOT_BLOCK_MIN_AREA_RATIO = 0.002f;
// 手持续多少帧才认为是长时间操作，长时间操作只提高整理证据权重
constexpr int HAND_LONG_PRESENT_FRAMES = 12;
// 手与物品重叠到什么程度，才作为 candidate_held 证据
constexpr float HELD_HAND_OVERLAP_THRESH = 0.30f;
// candidate_held 连续不可见多少帧后，建立 HELD 代理证据
constexpr int HELD_CONFIRM_FRAMES = 2;
// 物体 track 移动超过自身对角线多少比例，才作为移动证据
constexpr float TRACK_MOVE_DISTANCE_RATIO = 0.50f;

// =========================================================================
//  新业务流程7：出库物品过期
// -------------------------------------------------------------------------
//  OUT 状态的物品超过此时间仍未重新出现 → 确认出库，从记录中清除
// =========================================================================
constexpr long long OUT_ITEM_EXPIRE_MS = 600000LL;  // 10分钟

// =========================================================================
//  帧尺寸（用于手部扩展等计算）
// =========================================================================
constexpr float FRAME_W = 1280.0f;
constexpr float FRAME_H = 720.0f;

}  // namespace fridge

#endif  // __FRIDGE_CONFIG_H
