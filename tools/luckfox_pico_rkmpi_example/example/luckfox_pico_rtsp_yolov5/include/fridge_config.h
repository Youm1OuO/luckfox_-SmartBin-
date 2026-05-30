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
