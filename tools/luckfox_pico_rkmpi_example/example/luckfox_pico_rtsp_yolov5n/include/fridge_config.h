// ============================================================================
//  fridge_config.h
//  冰箱视觉系统 - 全局配置
//  在这一个文件里集中所有"魔术数字"和类别 ID 配置，方便后续调参。
// ============================================================================
#ifndef __FRIDGE_CONFIG_H
#define __FRIDGE_CONFIG_H

namespace fridge {

// 每次会话状态机行为有实质调整时递增。用于板端启动日志核实真正运行的二进制。
constexpr const char* FLOW3_BUILD_TAG = "3.0-r21";

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
constexpr float YOLO_OBJECT_SCORE_THRESH    = 0.40f;    // 物品的通用阈值
constexpr float YOLO_HAND_SCORE_THRESH      = 0.30f;    // 手的阈值

// ---- 单类特殊阈值（比通用物品阈值更高，用于压掉特定误判）----
//  背景：透明鸡蛋盒空出来后会被误判成 milk_box(cls_id=21)，误判分数多在 60~76；而真实
//  milk_box 正立摆放时通常 80+。所以给 milk_box 单独订一个更高的入库门槛 0.77，把空盒的
//  误判(≤0.76)挡在业务层之外。代价：真 milk_box 若“躺放”分数可能跌到 60 而被挡——演示时
//  正立摆放 milk_box 即可（已与用户确认）。想调回通用阈值，把此值改成 0.40 即可。
constexpr int   MILK_BOX_CLS_ID          = 21;      // milk_box 的 cls_id（与训练 label 对齐）
constexpr float MILK_BOX_SCORE_THRESH    = 0.77f;   // milk_box 专用入库门槛（高于通用 0.40）

inline float yolo_output_score_threshold(int cls_id) {
    if (is_hand(cls_id)) return YOLO_HAND_SCORE_THRESH;
    if (cls_id == MILK_BOX_CLS_ID) return MILK_BOX_SCORE_THRESH;
    return YOLO_OBJECT_SCORE_THRESH;
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
        case 43: case 44: case 45: case 46: case 47: case 48:
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
        "秋葵", "白菜", "豆角", "生菜", "卷心菜", "苦瓜", "节瓜"
    };
    // cn_names 的元素个数必须与 model/labels_list.txt 的行数一致。
    // 新增类别时：labels_list.txt 末尾加英文名，这里 cn_names 末尾加对应中文即可，
    // 边界 CN_NAMES_COUNT 用 sizeof 自动计算，无需手改。
    constexpr int CN_NAMES_COUNT = (int)(sizeof(cn_names) / sizeof(cn_names[0]));
    if (cls_id >= 0 && cls_id < CN_NAMES_COUNT) return cn_names[cls_id];
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
//    export FRIDGE_CLOUD_HOST=192.168.5.109
//    export FRIDGE_CLOUD_PORT=8000
//  上报端点：
//    POST /api/v1/admin/device-ingest      单个事件 ITEM_IN/OUT/MOVED
//    POST /api/v1/admin/events/heartbeat    设备心跳（每30秒）
//  鉴权：所有接口均需 Authorization: Bearer <token> 请求头
// =========================================================================
// 板子经 USB 网络(eth0, 192.168.168.x)访问宿主机 PC；后端绑 0.0.0.0，
// 从 PC 的 USB 网卡地址 192.168.168.1:8000 即可访问到(实测两个网卡都返回200)。
// 板子若改为同网段直连后端，用环境变量 FRIDGE_CLOUD_HOST=192.168.5.109 覆盖。
constexpr const char* CLOUD_HOST       = "192.168.168.1";     // USB 对端 PC 地址（实测可连后端）
constexpr int         CLOUD_PORT       = 8000;                // 端口
constexpr const char* CLOUD_ITEM_PATH  = "/api/v1/admin/device-ingest"; // 出入库事件端点
constexpr const char* CLOUD_HEARTBEAT_PATH = "/api/v1/admin/events/heartbeat"; // 心跳端点
constexpr const char* CLOUD_LOGIN_PATH = "/api/v1/admin/auth/login"; // 登录端点
// [已废弃] 后端无 /inventory/close 端点（实测 404）。关门整柜快照已停用，
// 库存完全依赖逐帧 ITEM_IN/OUT/MOVED 维护。保留常量仅为兼容旧签名，不再被使用。
constexpr const char* CLOUD_INVENTORY_PATH = "/api/v1/admin/inventory/bulk"; // 保留：整柜批量库存端点（当前未使用）
constexpr const char* CLOUD_DEVICE_ID  = "luckfox-001";       // 设备 ID
constexpr const char* CLOUD_USERNAME   = "admin";              // 登录用户名
constexpr const char* CLOUD_PASSWORD   = "181511";             // 登录密码（实测有效）
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
// Dedicated threshold for DISAPPEARANCE_SUPPORTED.  It does not participate
// in identity, D/MOVED confirmation, ordinary OUT, or blocker creation.
constexpr float FLOW3_CONFIRMED_OCCLUSION_DISAPPEARANCE_MIN_COVER_RATIO = 0.85f;
// Causal front-missing may accept a visibly partial front only when the
// overlap is still substantial enough to be a plausible blocker.  This lower
// bound applies only with unresolved HAND/POSSIBLE_MOVED target evidence; it
// does not weaken ordinary geometric disappearance or OUT.
constexpr float FLOW3_CAUSAL_OCCLUSION_MIN_COVER_RATIO = 0.30f;
// 3.0 的 D→C 遮挡在 D 已确认放下时才写入 block_ids。新 D 只部分覆盖
// C 但 C 已处于 HAND_* 时，达到该比例可以作为“先保留 C，不判 OUT”的证据。
constexpr float FLOW3_D_PARTIAL_COVER_RATIO = 0.30f;

// 可开关的 3.0 状态机诊断追踪。开启后会按操作号、帧号记录状态转换、
// C->B 仲裁、D 防线和无手结算依据；它只输出日志，不参与任何业务判断。
constexpr bool FLOW3_DEBUG_TRACE_LOG = false;

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
// [已废弃] 曾用于按置信度给业务框换色（低分橙色）。因为颜色跳变容易被误认为
// 识别出错，现已改为"一类一色、与置信度无关"，此阈值不再被引用。保留定义仅为
// 历史参考，可安全删除。
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
// 业务层维护 hand_id。以下参数只限制旧手轨迹与当前手框的身份恢复，
// 不改变 HAND_* 覆盖率、物品移动或无手结算阈值。稳定的一对一连续帧
// 正常累计位移；几何突变、合并或身份不可靠时暂停而不猜测。
constexpr float HAND_TRACK_MATCH_MAX_CENTER_NORM = 3.00f;
constexpr float HAND_TRACK_MATCH_MAX_SHAPE_DELTA = 1.20f;
constexpr float HAND_TRACK_MATCH_MAX_AREA_RATIO = 4.00f;
// 最优和次优全局分配的总代价差不超过此值时，身份不可靠，暂停该帧 delta。
constexpr float HAND_TRACK_MATCH_AMBIGUITY_MARGIN = 0.20f;
// 某条手轨迹在此数量的有手帧内没有唯一匹配时，保留其历史供恢复；期间
// 不向关联物品追加位移。超过窗口后丢弃过期 hand_id，等待重新建立。
constexpr int HAND_TRACK_TEMP_LOST_FRAMES = 2;
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

// =========================================================================
//  YOLO 输出后的手动纠正 (reclassify)
// -------------------------------------------------------------------------
//  背景：egg/orange 外形接近、apple/onion 颜色接近，未重训模型前在这里对
//  YOLO 的 cls_id 做基于"框面积"和"框内颜色"的启发式纠正。所有阈值集中在此，
//  调参只改这里，不用动 reclassify.cc 的逻辑。
//
//  相关类别 ID（必须与 model/labels_list.txt 行号一致，0 起）：
//    apple=0  orange=2  onion=15  egg=18  chinese_cabbage=43  lettuce=45
// =========================================================================

// 总开关：置 false 可完全关闭纠正层，回到 YOLO 原始输出（排查问题时用）。
constexpr bool RECLASSIFY_ENABLED = true;

// ---- 相关类别 ID（若 labels_list.txt 行号变化，改这里即可） ----
constexpr int CLS_APPLE           = 0;
constexpr int CLS_ORANGE          = 2;
constexpr int CLS_ONION           = 15;
constexpr int CLS_EGG             = 18;
constexpr int CLS_CHINESE_CABBAGE = 43;
constexpr int CLS_LETTUCE         = 45;

// ---- egg <-> orange：按"框面积占整幅画面比例"判断 ----
//  逻辑：egg 框够大 → 其实是 orange；orange 框够小 → 其实是 egg。
//  用面积比例(相对 FRAME_W*FRAME_H)而非绝对像素，物体远近更稳。
//  例：0.020 约等于 1280*720 的 2%，即约 18432 像素²。
//  分成两个阈值，避免同一物体在边界处来回翻。建议 EGG->ORANGE 略大于
//  ORANGE->EGG，留一段"谁都不改"的缓冲区。
constexpr float RECLS_EGG_TO_ORANGE_AREA_RATIO = 0.020f;  // egg 面积比 > 此值 → 改 orange
constexpr float RECLS_ORANGE_TO_EGG_AREA_RATIO = 0.015f;  // orange 面积比 < 此值 → 改 egg

// ---- apple <-> onion：按框内区域的平均颜色(HSV)判断 ----
//  逻辑：apple 偏红偏亮 → 保持；若偏紫偏暗 → 改 onion。
//        onion 偏紫偏暗 → 保持；若偏红偏亮 → 改 apple。
//  采样：取框内中心区域(去掉边缘背景)转 HSV，算平均 H(色相) 和 V(亮度)。
//  OpenCV 的 H 范围是 0~179，V 范围 0~255。
//  "偏暗"：V < RECLS_DARK_V_MAX。"偏亮"：V > RECLS_BRIGHT_V_MIN。
//  "偏紫"：H 落在 [PURPLE_H_MIN, PURPLE_H_MAX]（品红/紫，约 135~170）。
//  "偏红"：H 落在红区间（约 <=10 或 >=170，红在 HSV 环两端）。
constexpr float RECLS_DARK_V_MAX    = 55.0f;    // 平均亮度低于此 → 判为"暗"
constexpr float RECLS_BRIGHT_V_MIN  = 60.0f;   // 平均亮度高于此 → 判为"亮"
constexpr float RECLS_PURPLE_H_MIN  = 130.0f;   // 紫/品红色相下界
constexpr float RECLS_PURPLE_H_MAX  = 170.0f;   // 紫/品红色相上界
constexpr float RECLS_RED_H_LOW_MAX = 12.0f;    // 红色相下段上界 (H<=此值算红)
constexpr float RECLS_RED_H_HIGH_MIN= 168.0f;   // 红色相高段下界 (H>=此值算红)
// 颜色采样时框向内收缩的比例(各边)，避免采到背景。0.2 表示每边缩 20%。
constexpr float RECLS_COLOR_INSET_RATIO = 0.20f;

// ---- 假 orange 过滤（细节：两个 egg 拼成的长框会被误判成 orange，分数多在 60 左右）----
//  判据（三条同时满足才【删框】，宁可漏过、不可误删真 orange）：
//    1) 是 YOLO 原生 orange（cls_id==CLS_ORANGE）；
//    2) 分数 < ORANGE_FILTER_SCORE_THRESH；
//    3) 框内平均色相【不在橙色区间】(不橙)。
//  必须在 egg<->orange 面积互转【之前】执行，只作用于 YOLO 原生 orange，绝不碰
//  由 egg 转来的 orange，从而不会间接误删 egg。egg<->orange 转换本身保持纯面积、不加颜色。
constexpr bool  RECLS_ORANGE_FILTER_ENABLED   = true;   // 开/关这个假 orange 过滤
constexpr float RECLS_ORANGE_FILTER_SCORE_MAX = 0.63f;  // 分数 < 此值 才可能被过滤
constexpr float RECLS_ORANGE_H_MIN = 8.0f;   // 橙色色相下界(HSV,0~179)；H 在 [MIN,MAX] 算“橙”
constexpr float RECLS_ORANGE_H_MAX = 25.0f;  // 橙色色相上界；落在区间内=橙(真橙子)，区间外=不橙

// ---- 类别归一化(合并难区分的类) ----
//  把某个类统一记成另一个代表类。下面是一张“合并表”，你可以【只改这张表】随意增删合并
//  关系，不用改任何代码：
//    · 每行 { from, to } 表示“把 from 类改记成 to 类”（from 被替换掉、统一成 to）。
//      例：{ CLS_CHINESE_CABBAGE, CLS_LETTUCE } = 把白菜并入生菜。
//    · 要新增合并：加一行即可。要停用某条：把那行注释掉即可。
//    · from / to 必须是已定义的 CLS_xxx 常量。
//  ⚠️ 填表规则(照做即安全，避免误合并)：
//    1) from 不要等于 to（自己并自己没意义）。
//    2) 同一个 from 只写一行（别让一个类有两个去向）。
//    3) 不要写“链式”：例如别同时写 A→B 和 B→C。合并只做一轮、不连锁，
//       写成链式不会连续替换，只会按各自那行独立套用，容易与预期不符。
struct ClassMerge { int from; int to; };
constexpr ClassMerge RECLS_CLASS_MERGES[] = {
    { CLS_CHINESE_CABBAGE, CLS_LETTUCE },   // 白菜 → 生菜（把白菜统一记成生菜）
    // 以后想加就照上面加行(使用前先自己在本文将开头声明对应的变量并对其他的索引)，例如：
    // { CLS_XXX, CLS_YYY },
};
constexpr int RECLS_CLASS_MERGES_COUNT =
    static_cast<int>(sizeof(RECLS_CLASS_MERGES) / sizeof(RECLS_CLASS_MERGES[0]));

// ---- 纠正后的同类去重(补做一次 NMS) ----
//  背景：YOLO 的 NMS 是"逐类"进行的，只压制同类重叠框。所以 YOLO 会对同一个
//  物体同时给出 apple 框和 onion 框（cls_id 不同，互不压制）。我们把 onion 改成
//  apple 后，同一位置就出现两个 cls_id 相同、且高度重叠的框，而 NMS 早已在
//  postprocess 阶段跑完，不会再清理。这里在纠正之后补做一次同类去重。
//
//  判据用 IoM(交集/较小框面积)而非纯 IoU：一个框基本被另一个框包住时 IoM 更敏感，
//  正是"同一物体两个框"的典型形态。IoM 超过阈值且两框同 cls_id，只保留分数更高的。
//
//  安全边界(做法 A)：只在"这一对框里至少有一个是被 reclassify 改过类别的"时才去重，
//  绝不碰 YOLO 原生就给出的同类框——那是 YOLO 自己的 NMS 已放行的，二次删除可能误伤
//  两个真实相邻的同类物体(如挨着的两个苹果)，造成漏检。
constexpr bool  RECLS_DEDUP_ENABLED   = true;   // 是否启用纠正后同类去重
constexpr float RECLS_DEDUP_IOM_THRESH = 0.75f; // IoM 超过此值且同类，去重(保留高分框)

// =========================================================================
//  无手期后手矫正（补登记）—— 细节31
// -------------------------------------------------------------------------
//  一次操作(开门→有手→无手)结束时，对"有手期因证据不足没能定论"的 IN/OUT 做
//  一次严格准入的兜底。典型场景：透明蛋盒整盒放入/取走(YOLO 只看到手和鸡蛋)。
//  核心钥匙：候选必须落在"本轮手实际活动过的区域"内——手没去过的地方一律不补。
//  背景假框交给 YOLO 侧过滤，业务层相信 YOLO。有手期逻辑完全不受影响，纯增量。
// =========================================================================
// 总开关。置 false 立即回到"无手不补 IN/OUT"的原行为(排查/现场兜底用)。
constexpr bool  FLOW3_NOHAND_CORRECTION_ENABLED = true;
// 补 IN(连续稳定) / 补 OUT(连续缺失) 都要求的帧数。3 比 2 更能排除双帧幻影。
constexpr int   FLOW3_NOHAND_CORRECT_FRAMES = 3;
// "手活动区域"外扩比例：每个手框按自身宽/高各边外扩此比例(0.20=宽高各放大约40%)。
// 用比例而非固定像素，手离相机远近都稳。放大覆盖盒边物品，过大则纳入过多背景。
constexpr float FLOW3_NOHAND_CORRECT_HAND_MARGIN_RATIO = 0.20f;
// 候选框与"手活动区域"的重叠判据(IoM)。>= 此值算"大部分落在手动过的地方"。
// 补 IN 和补 OUT 用同一把尺。
constexpr float FLOW3_NOHAND_CORRECT_REGION_IOM = 0.50f;
// 手离开后无手后处理的封顶帧数（细节31 v3 闸门二，防 0/1 震荡死循环）。has_unresolved
// 持续为真、累计超过此帧数就强制让无手纠正接管。默认 3：手离开 3 帧还给不出结果，后续也
// 不会有；正好与无手纠正的 3 帧确认重叠。调大=给正常收敛更多余地；调小=更快接管但可能
// 打断需要多帧的正常场景。
constexpr int   FLOW3_NOHAND_POSTPROCESS_MAX_FRAMES = 3;

}  // namespace fridge

#endif  // __FRIDGE_CONFIG_H
