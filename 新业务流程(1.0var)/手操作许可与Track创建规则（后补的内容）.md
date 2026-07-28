# 手操作许可与 Track 创建规则

这份规则补在原有“库存 + 稳定快照 + Track”流程的最外层。

目的很简单：**没有检测到手，就不允许 YOLO 抖动直接改正式库存；检测到手，也不能把手附近全部物品都建成 Track。**

---

## 1. 手操作许可：只有本轮出现过手，才允许改库存

使用一个布尔值：

```cpp
bool operation_pending = false;
```

它的含义是：**从上一次库存结算后，到现在为止，是否检测到过一次手操作。**

```text
开门 / 上一次结算刚结束：
    operation_pending = false

某一帧检测到手：
    operation_pending = true
    清空正在收集的无手快照帧

手离开后，收集到 N 帧连续稳定无手画面：
    如果 operation_pending = true：
        允许这份稳定快照与库存比较
        可以确认：放入 / 取出 / 整理 / 遮挡 / 露出
        结算结束后 operation_pending = false

    如果 operation_pending = false：
        不允许新增、取出、整理、遮挡等正式库存变化
        只可刷新已正常匹配物品的 last_seen_box
```

### 例子

```text
第 1~4 帧：无手，库存稳定
    operation_pending = false

第 5 帧：YOLO 漏检一个苹果，但没有手
    不允许 OUT，只等待后续画面

第 6 帧：苹果重新被识别到
    不允许 IN，因为本轮没有手操作

库存不变。
```

> 例外：开门后的第一份稳定快照仍可用于初始化库存；这不是一次普通的出入库结算。

---

## 2. 只有确认发生移动，才创建 Track

**“手在物品附近”只能说明可能会操作它，不能说明它已经移动。**

因此分两层：

```text
Candidate（候选）
    手碰到 / 覆盖某个库存物品时，临时记住它。
    还不是 Track，不会用于整理判断。

Track（正式轨迹）
    只有确认物品或手代理真的移动后，才从 Candidate 升级而来。
```

### 2.1 手遮住一部分、YOLO 仍识别到物品

```text
手靠近苹果，YOLO 仍能识别苹果：
    先绑定 Candidate 到这个 item_id

下一帧苹果框相对上一帧苹果框移动足够明显：
    创建正式 Track
    Track.path 开始记录估计框

苹果框没有明显移动：
    只保留 Candidate
    不创建 Track
```

### 2.2 手完全遮挡物品、YOLO 识别不到物品

```text
苹果在手覆盖前还可见；随后苹果框消失，且手确实覆盖苹果原位置：
    建立“完全遮挡 Candidate”
    记录：苹果原位置、手的起始框

之后手框相对起始框移动足够明显，且原位置依旧没有重新检测到该物品：
    Candidate 升级为正式 Track
    Track.proxy_box 跟随手的位移移动
    Track.path 开始记录代理框

手移开后，原位置又检测到该物品：
    说明物品没有被拿走，丢弃 Candidate
    不创建 Track

手一直没移动就离开：
    丢弃 Candidate
    不创建 Track
```

这样可以处理“物品完全被手挡住后被移动”，同时不会因为手只是停在旁边，就给所有附近物品建立轨迹。

### 2.3 手离开后

```text
连续 N 帧无手稳定快照形成：

    有 Track，且旧物品消失、同类新物品出现在 Track 的放下位置：
        判为【整理】

    没有 Track 或 Track 证据不够：
        不猜测整理
        按放入 / 取出 / 遮挡的正常流程处理

    Candidate 但从未升级为 Track：
        直接丢弃
```

---

## 3. 三个小例子

### 例子 A：手进来，但没有移动任何物品

```text
手出现 → operation_pending = true
所有物品框都没有明显移动 → 只有 Candidate，没有 Track
手离开 → 生成稳定快照，库存没有变化
结算结束 → operation_pending = false，丢弃所有 Candidate
```

### 例子 B：移动一罐苏打水

```text
手出现 → operation_pending = true
苏打水框相对上一帧移动 → Candidate 升级为 Track#1
手离开 → 稳定快照中旧位置少了苏打水、新位置多了苏打水
Track#1 的放下位置与新位置匹配 → 判为【整理】
结算结束 → operation_pending = false，清空 Track#1
```

### 例子 C：无人操作时 YOLO 漏检苹果

```text
operation_pending = false
苹果突然漏检 → 不允许 OUT
苹果重新出现 → 不允许 IN
只等待下一份正常稳定快照刷新 last_seen_box
```

---

## 4. 与原有设计的关系

这两个规则不是替换原有设计，而是给原有设计加两道门：

```text
检测到手？
    否 → 不允许正式库存变化
    是 → 本轮允许在稳定快照时结算

物品真的移动？
    否 → 不创建 Track
    是 → 创建 Track，为“整理”提供证据
```

仍然保留：稳定快照、`anchor_box`、`last_seen_box`、Track 路径、遮挡处理和 `planned_changes`。

---

## 5. 快照与库存的最终确认（单向握手）

这一步只在：

```text
operation_pending = true
```

的一次操作结算中执行；位置在 `planned_changes` 提交正式库存之前。

本轮结算时，临时记录：

```text
当前稳定快照第 0 个物品 → item#2
当前稳定快照第 1 个物品 → item#5
当前稳定快照第 2 个物品 → item#8
```

每个快照物品都必须有唯一的 `item_id`：

```text
普通匹配成功      → 绑定旧 item_id，状态为 VISIBLE
整理匹配成功      → 绑定被移动的 item_id，状态为 VISIBLE
遮挡物重新可见    → 绑定原 item_id，状态为 VISIBLE
真正新物品        → 新建 item_id，状态为 VISIBLE
```

最后检查：

```text
快照中每一个物品
    都必须绑定到库存中唯一一个 VISIBLE 的 item_id
```

注意：最后检查不能重新按“同类、距离近”猜一次对应关系；只能使用本轮前面已经确定的“快照物品 → item_id”绑定。

反方向不强制：库存中的一个 VISIBLE 物品暂时没出现在快照中，可能只是 YOLO 没识别到，不能因此立刻把它取出。

### 握手失败时怎么处理

```text
检查通过：
    提交 planned_changes

检查失败：
    跳过本次库存修改
    正式库存保持不变

也就是说,"单向握手"只是一个检查, 并不强制拉item_id, 如果检查通过了旧修改库存, 如果不通过就跳过本次库存修改
```
