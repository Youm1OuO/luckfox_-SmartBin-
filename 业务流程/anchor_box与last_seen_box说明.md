# `anchor_box` 与 `last_seen_box` 说明

这两个框都是库存物品的长期视觉信息，应和 `item_id`、`cls_id`、`status` 一起保存到库存表，并在快照对比确认后同步给后台。

```cpp
struct InventoryItem {
    int item_id;
    int cls_id;
    ItemStatus status; // VISIBLE / OCCLUDED / OUT

    bool anchor_valid;
    BBox anchor_box;
    BBox last_seen_box;
    // status == OCCLUDED 时用的计数器
    int no_occluder_count;
    int duplicate_count;
};
```

## 1. `anchor_box`：可靠参考框

`anchor_box` 表示这个物品较可靠的参考位置和尺寸，尽量接近物品未被遮挡时的完整框。

它主要用于：

- 手刚开始移动物品时，作为 Track 的起始参考框；
- 物品被手完全遮挡后，估计物品跟随手移动时保持原本尺寸；
- 判断遮挡物品附近是否重新出现物品；
- 快照确认“整理”后，更新该物品的新位置参考。

新物品刚放入、仍被手挡住时，当前检测框可能只是局部框，此时不要把它当作可靠参考：

```cpp
item.last_seen_box = 当前局部检测框;
item.anchor_box.reset();  // 尚无可靠参考框
```

在手离开后的第一份稳定快照中，若该物品的框足够可靠，则：

```cpp
item.anchor_box = 当前稳定框;
item.last_seen_box = 当前稳定框;
```

`anchor_box` 一旦建立，不应因为普通的局部遮挡或 YOLO 抖动而覆盖；但物品被确认整理到新位置后，应更新为新位置的可靠参考框。

## 2. `last_seen_box`：最近实际看到的框

`last_seen_box` 是 YOLO 最近一次实际检测到该物品时的框。它允许是局部框。

```text
物品完整可见：last_seen_box = 完整检测框
物品只露出一半：last_seen_box = 半个物品的检测框
物品完全被遮挡：last_seen_box = 上一次看到它时的框
```

它主要用于把当前帧检测结果关联到已有物品，以及记录最近一次真实观测结果。


## 3. 手部移动时如何使用

每一帧都要处理 YOLO 结果和手；只有正式库存修改要等待稳定快照确认。

手开始移动一个物品时：

```cpp
if (item.anchor_box.has_value()) {
    track.start_box = *item.anchor_box; // 有可靠完整参考
} else {
    track.start_box = item.last_seen_box; // 只能暂时使用局部框
}
```

所以，没有 `anchor_box` 时仍然可以建立和记录 Track；只是不能把局部框误认为完整物品的可靠尺寸。后续“整理”判定应更保守。

## 4. 与后台、Track 的边界

```text
后台 / InventoryDB：item_id、cls_id、status、anchor_box、last_seen_box、计数器
本地 ProcessStore：当前帧结果、手框、稳定快照缓冲、Track 路径
```

Track 只用于一次开门过程内判断“是否整理”，不传给后台，快照确认并提交库存后可清空。

库存表也不应每帧同步后台；应在稳定快照对比完成、库存结果确认后，提交更新后的库存项或变更项。

## 5. 坐标要求

两个框必须始终使用同一套坐标：旋转后的原始摄像头图像坐标。后台保存和返回时也必须保持同一坐标约定；若以后调整分辨率、旋转方式或裁剪方式，需要加入坐标版本或迁移旧框。
