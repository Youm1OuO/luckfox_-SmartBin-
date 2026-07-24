# 重写计划：新业务流程5 核心逻辑

## 一、现有代码问题分析

逐一对比《新业务流程5.md》设计文档与现有代码，发现以下关键BUG：

### BUG 1：快照对比逻辑完全错误（最严重）

**设计要求**：对比两份稳态快照 snap1 vs snap2 的差异
**现有代码**：`compare_snapshots()` 第一步检查"库存中 VISIBLE 物品是否在 snap2 中"

具体问题（`session.cc:331-429`）：
- 第一步遍历的是 `inventory_` 中的 VISIBLE 物品，而不是从 snap1 中找出"在 snap1 但不在 snap2"的物品
- 库存经过多次操作后已经和 snap1 不一致了
- 结果：物品被错误标记为【出库】或【遮挡】

### BUG 2：快照状态机缺少变量

**设计要求**：4个变量 `snap1/contrast/current/snap2`
**现有代码**：只有 `snap1_`/`contrast_`/`current_`，没有 `snap2_`

现有代码在 `snapshots_similar()` 返回 true 时直接对比 snap1 vs current，跳过了 snap2 的显式记录。虽然语义上勉强能工作，但不够清晰。

### BUG 3：snapshots_similar() 对比条件可能太严格

**设计要求**：判断"大部分物品匹配上了"
**现有代码**：`match_ratio >= 0.8f && count_diff <= 2`

问题：`match_strict()` 使用 5 条件（含颜色），但两份快照间的物品颜色几乎不变，IoU=0.7 + 面积比=0.2 + 中心距离=15px 的严格几何条件在稳定场景中已经足够。颜色检查增加了不必要的失败风险（如亮度微小变化）。

### BUG 4：初始化匹配用 cls_id 做去重键（错误）

**代码位置**：`session.cc:211` `if (matched_backend.count(vi.cls_id)) continue;`

问题：用 cls_id 做去重键意味着同类别只能匹配一个物品。如果冰箱里有两个苹果，第二个永远不会被匹配上，会被错误标记为【遮挡】。

应该用 `VotingItem` 的索引做去重键。

### BUG 5：手离开后缺少稳定化等待

**设计要求**：手离开后再等 3 帧（`HAND_STABLE_FRAMES`）再拍快照对比
**现有代码**：手一离开就立即处理（`session.cc:168-191`），没有等待

### BUG 6：手部覆盖检测阈值过低

**代码位置**：`session.cc:131` `overlap_ratio_of_smaller(hand_box, kv.second.box) > 0.0f`

阈值 0.0f = 任何像素级重叠都算"被手抓住"。应该使用 0.3f。

### BUG 7：主循环流水线顺序有误

**设计要求**：ByteTrack 每帧都跑 → 手不在时推快照缓冲 → 手在时走手部逻辑
**现有代码**：先跑手检测再推快照

流水线顺序应为：
1. ByteTrack 更新
2. 推入快照缓冲
3. 手检测 + HELD 逻辑
4. 快照对比（无手时）

### BUG 8：手离开后的完整流程缺失

**设计要求**（文档第九点）：
1. 手离开 → 立即重新检查因手而变为【遮挡】的物品
2. 等待稳定（3帧）
3. 拍新快照 vs 手进入前快照
4. 判断拿走/放下/整理

**现有代码**：只做了第1步，缺少第2-4步。

---

## 二、重写范围

| 文件 | 操作 | 说明 |
|------|------|------|
| `session.h` | **重写** | 新增手部离开后的快照对比流程接口 |
| `session.cc` | **重写** | 修复快照对比三步逻辑 + 手离开流程 |
| `main.cc` | **小改** | 调整流水线顺序 |
| `snapshot.cc` | **微调** | 基本正确，不做大改 |
| `inventory.cc` | **微调** | 基本正确，修复初始化匹配键 |
| `fridge_config.h` | **不动** | 阈值配置已正确 |
| `geometry.h` | **不动** | 几何工具已正确 |
| `tracker.h/cc` | **不动** | ByteTrack 已正确 |

---

## 三、新 session.h 设计

```cpp
class SessionManager {
public:
    // 主接口（不变）
    void init_from_backend(const std::vector<InventoryItem>& items);
    SettlementResult update_hand(const std::vector<BBox>& hand_boxes,
                                 const std::vector<Track>& tracks,
                                 int frame_id, long long time_ms);
    SettlementResult push_snapshot(const Snapshot& snap, const cv::Mat& frame);

    // 查询接口（不变）
    bool has_backend() const;
    const InventoryDB& inventory() const;
    InventoryDB& inventory();
    bool hand_present() const;
    int no_hand_streak() const;

private:
    // ---- 快照状态机（4变量） ----
    SnapState snap_state_;
    Snapshot snap1_;          // 基准快照（用于对比）
    Snapshot contrast_;       // 稳定性检测基准
    Snapshot current_;        // 当前快照
    bool has_snap1_;

    // ---- 手部状态（新增：手进入前快照） ----
    bool hand_present_;
    int no_hand_streak_;
    std::set<int> hand_occluded_ids_;
    Snapshot snap_before_hand_;   // 新增：手进入前的最后一份稳态快照
    bool has_snap_before_hand_;

    // ---- 库存 ----
    InventoryDB inventory_;
    bool backend_initialized_;
    std::vector<InventoryItem> backend_items_;
    long long current_time_ms_;

    // ---- 辅助方法 ----
    void init_snapshot(const Snapshot& snap, const cv::Mat& frame);
    bool snapshots_similar(const Snapshot& a, const Snapshot& b);  // 无frame参数
    SettlementResult compare_snapshots(const Snapshot& snap2, const cv::Mat& frame);
    void process_post_hand_snapshot(const Snapshot& snap, const cv::Mat& frame);
    bool match_strict(const BBox& a, int cls_a, const BBox& b, int cls_b, const cv::Mat& frame);
    float color_diff(const cv::Mat& frame, const BBox& a, const BBox& b);
    void print_inventory();
};
```

---

## 四、核心算法重写细节

### 4.1 compare_snapshots() 三步对比（核心修复）

**第一步**：snap2 比 snap1 少的物品
```
for each item A in snap1:
    if A not in snap2 (4条件无颜色匹配):
        C = 在snap2中找A附近的物品 (normalized_nearby_distance < 1.2)
        if C 不存在:
            → A 被拿走, A 状态变【出库】
        if C 存在, 且 C 与库存中某个【遮挡】物品匹配(只比类别+中心点):
            → A 被拿走(【出库】), C 露出来(【可见】并更新bbox)
        if C 存在, 且 C 不在库存中:
            → C 是新物品压住了 A, A 变【遮挡】
        if C 存在, 且 C 在库存中(【可见】):
            → A 被拿走(【出库】)(C本来就在这里)
```

**第二步**：snap2 中所有物品与库存做严格匹配
```
for each item C in snap2:
    先匹配【出库】状态物品 → 匹配上则恢复为【可见】
    再匹配库存(非【出库】):
        匹配上 + 【遮挡】→ 变【可见】
        匹配上 + 【可见】→ 更新位置
        没匹配上 → 新物品入库
```

**第三步**：例行清理
```
清理冗余【出库】：与(【遮挡】/【可见】)匹配 → 删除
清理冗余【遮挡】：与【可见】匹配 → 删除
```

### 4.2 快照状态机

```
push_snapshot(snap, frame):
    if snap.has_hand:
        if !hand_present_:    # 手刚进入
            snap_before_hand_ = snap1_    # 记录手进入前的快照
        return

    if state == IDLE:
        init_snapshot(snap, frame)
        snap1_ = snap; contrast_ = snap
        state = COMPARE

    elif state == COMPARE:
        current_ = snap
        if snapshots_similar(current_, contrast_):
            result = compare_snapshots(current_, frame)
            snap1_ = current_    # 更新基准
            return result
        else:
            contrast_ = current_    # 更新对比基准，继续等稳定
```

### 4.3 手离开后的处理

```
update_hand(hand_boxes, tracks, frame_id, time_ms):
    if hand 存在:
        标记手覆盖的物品 → hand_occluded_ids_
        # 不做其他处理，等待手离开

    if hand 离开(no_hand_streak > 0):
        # 立即检查因手而变遮挡的物品
        for item_id in hand_occluded_ids_:
            if item 在当前检测中可见 → 恢复为【可见】
            else → 标记为【出库】并上报事件
        hand_occluded_ids_.clear()

        # 等待稳定后拍新快照对比
        if no_hand_streak == HAND_STABLE_FRAMES && has_snap_before_hand_:
            process_post_hand_snapshot(current_stable_snap, frame)
```

### 4.4 snapshots_similar() 修复

使用 4 条件（无颜色）做快照间稳定性判断：
- 类别相同
- 中心距离 < 15px
- 面积比差异 < 0.2
- IoU > 0.7

相似性判断：匹配率 >= 70% 且物品数量差异 <= 2

---

## 五、main.cc 流水线修改

```cpp
// 每帧：
1. YOLO 推理 + ByteTrack 更新
2. 推入快照缓冲区（含手标志）
3. 攒满3帧 → 取快照 → 送入 SessionManager::push_snapshot()
4. SessionManager::update_hand(hand_boxes, tracks, ...)
5. 处理事件 + 画面绘制
```

---

## 六、实施步骤

1. 重写 `session.h`（新接口 + 新成员变量）
2. 重写 `session.cc`（完整的新逻辑实现）
3. 修改 `main.cc`（调整流水线顺序）
4. 修改 `inventory.cc`（修复初始化匹配键）
5. 编译验证
