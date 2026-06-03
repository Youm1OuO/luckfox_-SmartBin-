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
//  上报端点：POST /events/item（事件 ITEM_IN/OUT/MOVED）。
//  放入(ITEM_IN) 带物品框内截图，后端拿到图后自行决定要不要跑云端 AI。
// =========================================================================
constexpr const char* CLOUD_HOST       = "192.168.168.1";   // 云端/网关主机
constexpr int         CLOUD_PORT       = 8000;              // 端口
constexpr const char* CLOUD_ITEM_PATH  = "/events/item";    // 出入库事件端点
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
//  手部状态机参数（hand_state.* 模块用，当前未在 main.cc 调用，保留备用）
// =========================================================================
// 手框与某个物品框的 IoU 高于此值视为"贴住"
constexpr float HAND_OBJ_IOU_HIGH = 0.40f;
// 手框与某个物品框的 IoU 低于此值视为"分离"
// (HAND_OBJ_IOU_HIGH > IoU > HAND_OBJ_IOU_LOW 是灰区，状态保持)
constexpr float HAND_OBJ_IOU_LOW  = 0.20f;

// 状态从"疑似"升级到"确定"需要连续保持的帧数（防抖动）
constexpr int CONFIRM_FRAMES = 3;

// 在确认"拿起"事件时，回看过去多少帧的物品类别做投票
// (拿起前的瞬间物品是完整可见的，识别更可靠)
constexpr int IDENTIFY_LOOKBACK = 5;


// =========================================================================
//  稳态切片 + 会话事件（stability / session 模块）
// -------------------------------------------------------------------------
//  这是当前 main.cc 实际使用的业务路径。核心思想：
//    把"事件"定义成"两个稳态之间发生了什么"，
//    每帧只判断"现在是稳态还是扰动态"，扰动态期间不更新库存，
//    扰动 → 稳态确认时，对前后两份 track 快照做 diff，得到一批事件。
// =========================================================================

// 进入"稳态"前需要连续多少帧"看起来稳"才算确认
// 10 FPS 下 6 帧 ≈ 0.6 秒，权衡延迟和稳定性
constexpr int STABLE_CONFIRM_FRAMES = 6;

// 一旦发现任何"扰动信号"立刻进入扰动态（不需要连续多帧），
// 因为漏判扰动态的代价（误判事件）远大于多扰动一次的代价（多等 0.6 秒）。
// 这里保留一个常量是为了未来若需要去抖动可以调，目前默认 1 帧即生效。
constexpr int DISTURBED_CONFIRM_FRAMES = 1;

// 物品 track 的中心点，过去多少帧位移在 STABLE_MOVE_PIX 之内才算"不动"
// 用于稳态判定的第二个条件（所有食材 track 都必须满足"不动"）
constexpr int STABLE_MOTION_WINDOW = 5;
// 物品 track 中心点位移阈值（像素），<= 该值视为"不动"
// 720x480 下 8 像素相当于物体中心点波动不超过 ~1%，足够松也足够严
constexpr float STABLE_MOVE_PIX = 8.0f;

// 整理事件的位置变化阈值：稳态前后同一 track_id 的中心点位移超过此值
// 才算是"挪过位"，否则当作"没动"。
// 比 STABLE_MOVE_PIX 大一些，避免抖动被当成 RELOCATE
constexpr float RELOCATE_MOVE_PIX = 30.0f;

// 进入稳态时所有 track 必须满足的最小本帧检测分数（防止 LOST 状态的 track
// 被误当作"还在画面里"）。低于此分数的 track 不计入稳态快照。
constexpr float SNAPSHOT_MIN_SCORE = 0.3f;

}  // namespace fridge

#endif  // __FRIDGE_CONFIG_H
