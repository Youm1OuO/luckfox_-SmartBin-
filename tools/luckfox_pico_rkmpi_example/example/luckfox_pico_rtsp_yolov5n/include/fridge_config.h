// ============================================================================
//  fridge_config.h
//  冰箱视觉系统 - 全局配置
//  在这一个文件里集中所有"魔术数字"和类别 ID 配置，方便后续调参。
// ============================================================================
#ifndef __FRIDGE_CONFIG_H
#define __FRIDGE_CONFIG_H

namespace fridge {

// 每次会话状态机行为有实质调整时递增。用于板端启动日志核实真正运行的二进制。
constexpr const char* FLOW3_BUILD_TAG = "3.0-r20";

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
//    3. 业务层手阈值：HAND_CONTEXT_SCORE_THRESH；检测到就停止无手收尾。
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
// 当前离线测试默认关闭登录、心跳和事件上传；接入后台后改为 true，或在运行时设置
// FRIDGE_CLOUD_ENABLED=1 覆盖本默认值。
constexpr bool        CLOUD_ENABLED    = false;

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
//  3.0 库存逐帧流程超参数
// -------------------------------------------------------------------------
//  这里的常量只给出可运行的初始值；部署前必须按该摄像头、该模型和实际
//  冰箱格局标定。详细的“地址 + 含义 + 调整方向”见：
//  另另一种业务流程(3.0var)/超参数位置与调节说明.txt
// =========================================================================

// 历史多帧投票模块参数。3.0 的 SessionManager 不使用 SnapshotBuffer，
// 这些常量只保留给独立的 snapshot.cc 兼容编译，不参与业务判断。
constexpr int   SNAPSHOT_N            = 3;
constexpr float SNAPSHOT_S            = 0.5f;
constexpr float SNAPSHOT_MIN_SCORE    = 0.5f;
constexpr float SNAPSHOT_MIN_BOX_WIDTH  = 12.0f;
constexpr float SNAPSHOT_MIN_BOX_HEIGHT = 12.0f;

// 同一份历史快照内跨帧投票候选的严格关联，不用于 3.0 库存结算。
constexpr float SNAPSHOT_VOTE_CENTER_NORM = 0.30f;
constexpr float SNAPSHOT_VOTE_WIDTH_RATIO = 0.25f;
constexpr float SNAPSHOT_VOTE_HEIGHT_RATIO = 0.25f;

// 库存 A 与快照 B 的“严格匹配”：类别相同 + 中心、宽、高均接近。
constexpr float INVENTORY_STRICT_CENTER_NORM = 0.26f;
constexpr float INVENTORY_STRICT_WIDTH_RATIO = 0.25f;
constexpr float INVENTORY_STRICT_HEIGHT_RATIO = 0.25f;

// “局部匹配”：类别相同且 IoM 接近 1。
constexpr float INVENTORY_PARTIAL_IOM = 0.84f;

// 3.0 身份/轨迹匹配。严格匹配用于稳定位置；轨迹匹配允许手持物体
// 的 YOLO 框略有抖动或局部遮挡，但仍要求类别和大致形状一致。
constexpr float FLOW3_TRACK_CENTER_NORM = 0.75f;
constexpr float FLOW3_TRACK_PARTIAL_IOM = 0.50f;
constexpr float FLOW3_TRACK_WIDTH_RATIO = 0.65f;
constexpr float FLOW3_TRACK_HEIGHT_RATIO = 0.65f;
// C 暂时不可见后，首次出现 B 的候选轨迹容差。它只用于把 B 暂存为
// reappear_candidate，不会单帧改变 item_id；下一帧仍需自匹配确认。
constexpr float FLOW3_REAPPEAR_CANDIDATE_CENTER_NORM = 1.25f;
constexpr float FLOW3_REAPPEAR_CANDIDATE_WIDTH_RATIO = 0.85f;
constexpr float FLOW3_REAPPEAR_CANDIDATE_HEIGHT_RATIO = 0.85f;
constexpr float FLOW3_OLD_POSITION_OVERLAP_AREA = 16.0f;

// 仅用于当前帧身份仲裁的跨类别重复提示。它不是全局检测过滤：只有低分框
// 与明显更高分的异类框几乎同框、且旧物已有可靠移动终点时，才暂时降低该
// 低分框对旧物原位身份的权威。普通异类重叠不会触发。
constexpr float FLOW3_CROSS_CLASS_DUPLICATE_LOW_SCORE_MAX = 0.45f;
constexpr float FLOW3_CROSS_CLASS_DUPLICATE_SCORE_GAP_MIN = 0.20f;
constexpr float FLOW3_CROSS_CLASS_DUPLICATE_IOM_MIN = 0.85f;
constexpr float FLOW3_CROSS_CLASS_DUPLICATE_CENTER_NORM_MAX = 0.16f;
constexpr float FLOW3_CROSS_CLASS_DUPLICATE_SIZE_RATIO_MAX = 0.20f;

// 物品刚被手挡住时，YOLO 输出的是完整框的一部分，不能继续用库存/快照的
// 高 IoM 阈值直接否定它。以下仅用于“当前手确实影响该框”的已有物品优先认领，
// 不用于普通无手库存匹配，也不单独生成身份。
constexpr float FLOW3_HAND_PARTIAL_MIN_OBSERVED_COVER = 0.30f;
constexpr float FLOW3_HAND_PARTIAL_MAX_AREA_RATIO = 1.35f;
constexpr float FLOW3_HAND_PARTIAL_CENTER_NORM = 0.85f;

// 动态矩形遮挡 / block_ids。
constexpr float BLOCK_OVERLAP_AREA_EPS = 4.0f;
constexpr float COVER_REMAINING_AREA_EPS = 4.0f;
// 严格覆盖只剩目标外边缘窄条带时的受限容差。它只在结算层确认了
// 当前 IN/MOVED blocker 后使用，不改变 COVER_REMAINING_AREA_EPS 的并集语义。
constexpr float FLOW3_CONFIRMED_OCCLUSION_EDGE_RESIDUAL_PX = 8.0f;
// 3.0 的 D→C 遮挡在 D 已确认放下时才写入 block_ids。新 D 只部分覆盖
// C 但 C 已处于 HAND_* 时，达到该比例可以作为“先保留 C，不判 OUT”的证据。
constexpr float FLOW3_D_PARTIAL_COVER_RATIO = 0.30f;

// 可开关的 3.0 状态机诊断追踪。开启后会按操作号、帧号记录状态转换、
// C->B 仲裁、D 防线和无手结算依据；它只输出日志，不参与任何业务判断。
constexpr bool FLOW3_DEBUG_TRACE_LOG = true;

// 当前尚未对接后台，允许首张无手直接检测建立本地测试库存。
// 接入可信后台后建议改为 false：此时冷启动画面只做只读校验，不负责建库。
constexpr bool ALLOW_INITIAL_FRAME_BOOTSTRAP_WHEN_BACKEND_UNAVAILABLE = true;

// =========================================================================
//  手框
// =========================================================================
// 手证据阈值：业务层只要检测到这种手框，就停止无手收尾。
constexpr float HAND_CONTEXT_SCORE_THRESH = YOLO_HAND_SCORE_THRESH;
// 强手框仅用于 OSD 状态提示，避免 UI 被普通小手框干扰。
constexpr float OSD_STRONG_HAND_SCORE_THRESH = 0.45f;
constexpr float OSD_STRONG_HAND_MIN_AREA_RATIO = 0.002f;
// 所有进入业务层的食品/手框都必须绘制；这个值只决定低置信度业务框的
// 视觉颜色，不再作为第二道显示过滤。
constexpr float OSD_LOW_CONFIDENCE_OBJECT_SCORE_THRESH = 0.50f;

// =========================================================================
//  3.0 HAND_* / Track 参数
// -------------------------------------------------------------------------
// 每个被手影响物品各自保存 move_values 和完整估计轨迹；正式事件仍在无手
// 逐帧收尾的条件提交时生成。对已有完整库存物品 A，先计算：
//   r = intersection(hand.box, A.box) / area(A.box)
// r < e2 不进入 HAND_*；e2 <= r < e1 为 HAND_PARTIAL；r >= e1 才为
// HAND_FULL。e2/e1 只使用 A 的完整可靠框，不能使用被遮挡后缩小的检测框。
// =========================================================================
constexpr float TRACK_HAND_MOVE_EPS          = 12.0f;
// 没有任何未决 HAND_* / CONTACT_* 轨迹时，连续两帧手框几乎没有变化才跳过。
// 只要有活动轨迹，即使手腕不动也必须继续检查物品框，避免漏掉手指推动。
constexpr float HAND_MICRO_MOVE_SKIP_EPS     = 6.0f;
// 疑似 D 连续有效观察次数达到该值后，认为身份稳定，可在结算时入库。
constexpr int   NEW_ITEM_CONFIRM_FRAMES      = 2;
constexpr int   FLOW3_HOLD_EVIDENCE_REQUIRED = 2;
constexpr int   FLOW3_NOT_HOLD_EVIDENCE_REQUIRED = 2;
// 无手阶段只累计同一对象的连续直接观测，绝不对多个检测框做投票或平均。
constexpr int   FLOW3_NO_HAND_D_CONFIRM_FRAMES = 2;
constexpr int   FLOW3_NO_HAND_OUT_MISSING_FRAMES = 2;
// 新建的旧库存 CONTACT_* / HAND_* 轨迹在后续两张有效帧内只能做本地
// 匹配和累计证据，不能把同类 B 作为排他归属；两张后续帧结束后才成熟。
constexpr int   FLOW3_NEW_TRACK_CLAIM_GRACE_FRAMES = 2;
constexpr float FLOW3_HAND_ATTACH_DISTANCE   = 28.0f;
constexpr float FLOW3_HAND_NEAR_MAX_INTERSECTION_AREA = 900.0f;
// e2：手至少覆盖完整物品面积的 30%，才把已有库存物品加入 HAND_*。
constexpr float FLOW3_HAND_PARTIAL_COVER_RATIO = 0.30f;
// e1：只有接近完全覆盖（当前 88%）才称为 HAND_FULL；中间均是 HAND_PARTIAL。
constexpr float FLOW3_HAND_FULL_COVER_RATIO  = 0.88f;
// 仅供“D 是否和手相贴/相交”和放下判断使用，不参与已有库存物品的 HAND 状态。
constexpr float FLOW3_HAND_DETECTION_OVERLAP_AREA = 300.0f;
// CONTACT_* 的实际物品框关联。中心容差比普通严格匹配宽，但仍要求类别和
// 宽高比例接近；它只用于低覆盖率推/拉，不用于普通库存匹配。
constexpr float FLOW3_CONTACT_PATH_CENTER_NORM = 1.10f;
constexpr float FLOW3_CONTACT_WIDTH_RATIO = 0.35f;
constexpr float FLOW3_CONTACT_HEIGHT_RATIO = 0.35f;
// 物品相对上一真实观测框至少移动多少，才算一条有效推/拉证据。
constexpr float FLOW3_CONTACT_OBJECT_MOVE_EPS = 12.0f;
// D 初次仅有局部框、手离开后重新出现完整框时的专用路径匹配。
// B 覆盖路径中预测局部框达到该比例，或中心偏移足够小，才允许局部→完整重现。
constexpr float FLOW3_D_REAPPEAR_MIN_PARTIAL_COVER = 0.35f;
constexpr float FLOW3_D_REAPPEAR_MAX_CENTER_SHIFT_NORM = 0.45f;
// 有手阶段：若一个无法认领的 D 覆盖了看不见的旧 C 的原位置达到该比例，
// 即使 D 已不贴手，也以 C_POSITION_REPLACEMENT_D 预登记它。
constexpr float FLOW3_C_REPLACEMENT_MIN_COVER_RATIO = 0.30f;
// 无手阶段：手离开后的前几张无手帧可从公共手轨迹创建
// POST_HAND_REVEAL_D。窗口内仍需一张后续直接帧完成自匹配。
constexpr int   FLOW3_POST_HAND_REVEAL_WINDOW_FRAMES = 2;
// 同类 B 首次重新出现后，需要多少次连续自匹配，才允许作为旧 C 的
// 已确认候选；最终仍由无手逐帧收尾的条件提交决定。
constexpr int   FLOW3_REAPPEAR_CANDIDATE_CONFIRM_FRAMES = 2;
constexpr float FLOW3_DROP_FALL_BEHIND_RATIO = 0.45f;
// 放下需要连续证据，不能只凭 B 与手脱离的一帧就确认。
constexpr int   FLOW3_DROP_EVIDENCE_REQUIRED = 2;
// B 的宽高相对完整 C 更接近多少，才可作为“逐渐露出完整框”的辅助放下证据。
constexpr float FLOW3_DROP_FULL_BOX_IMPROVEMENT = 0.08f;
// 最终 MOVED 判断使用 operation-start 的完整原框对角线作为尺度，并限制
// 极端框尺寸对归一化结果的影响。当前值只影响最终位置变化判断，不影响
// STRICT / LOCAL / CONTACT 的身份匹配几何范围。
constexpr float FLOW3_MOTION_REFERENCE_DIAGONAL_MIN_PX = 80.0f;
constexpr float FLOW3_MOTION_REFERENCE_DIAGONAL_MAX_PX = 500.0f;
// 归一化最终位移的静态参考线与正式移动线。两者之间是灰区，必须依赖
// 连续无手帧、唯一所有权和既有 HAND/CONTACT 证据，不能强制二选一。
constexpr float FLOW3_STATIC_CENTER_SHIFT_NORM = 0.06f;
constexpr float FLOW3_FORMAL_MOVE_CENTER_SHIFT_NORM = 0.12f;
// 兼容保留：旧版本的固定像素值不再承担最终 MOVED 判定职责；CONTACT
// 的短时物品移动仍使用 FLOW3_CONTACT_OBJECT_MOVE_EPS。
constexpr float FLOW3_COMMIT_MOVE_CENTER_DISTANCE = 28.0f;

// =========================================================================
//  门状态机（CLOSED / OPENING / OPEN / CLOSING）
// =========================================================================
constexpr float DOOR_OPENING_CANDIDATE_BRIGHTNESS = 58.0f;
constexpr float DOOR_OPEN_LIGHT_THRESHOLD = 68.0f;
constexpr int   DOOR_OPEN_CONFIRM_FRAME_COUNT = 5;
constexpr float DOOR_CLOSING_GUARD_THRESHOLD = 42.0f;
constexpr float DOOR_CLOSING_DROP_RATIO = 0.62f;
constexpr float DOOR_DARK_PIXEL_THRESHOLD = 38.0f;
constexpr float DOOR_CLOSING_DARK_PIXEL_RATIO_THRESHOLD = 0.72f;
constexpr int   DOOR_CLOSE_CONFIRM_FRAME_COUNT = 5;
constexpr float OPEN_REFERENCE_UPDATE_THRESHOLD = 72.0f;
constexpr float OPEN_DARK_PIXEL_RATIO_LIMIT = 0.35f;
constexpr int   OPEN_REFERENCE_WINDOW_SIZE = 20;

// =========================================================================
//  帧尺寸（用于手部扩展等计算）
// =========================================================================
constexpr float FRAME_W = 1280.0f;
constexpr float FRAME_H = 720.0f;

}  // namespace fridge

#endif  // __FRIDGE_CONFIG_H
