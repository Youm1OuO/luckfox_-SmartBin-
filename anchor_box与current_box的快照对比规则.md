# 用 anchor_box 与 current_box 处理大框 / 小框

我重新想后的结论：

> Track 只是“把 `OUT + IN` 升级成【整理】”的可选证据。
>
> 快照与库存的主流程不依赖 Track；Track 不存在、出错或不唯一时，直接退化成普通快照对比即可。

> 【出库】和【完整可见】/【局部可见】/【不可见】不是同一件事。
>
> 更清楚的做法是拆成两个字段，而不是把四种情况塞进一个 `status`。


# 1. 库存字段

```text
presence：IN / OUT；                   // 是否仍属于库存
view：FULL / PARTIAL / HIDDEN；         // 仅 presence == IN 时有效

anchor_box：最近一次确认的完整框；
current_box：最近一次确认看见的框；可大、可小；
```

```text
IN + FULL：
    current_box 接近 anchor_box；

IN + PARTIAL：
    current_box 是已确认的小框；

IN + HIDDEN：
    current_box 为空；anchor_box 仍保留；

OUT：
    current_box 为空；anchor_box 保留作历史参考。
```

> **“看得见”不等于一定是大框。**
>
> 物品只要成功绑定了快照 B，就是看得见；B 比 `anchor_box` 小时，`view = PARTIAL`。

> `current_box` 永远直接写稳定快照的 B.box，**不和旧框取平均**。
>
> `anchor_box` 只在【新放入】、【整理】、【放回】确认完整框时更新；普通匹配和局部匹配都不能缩小它。


# 2. 三种框匹配

```text
CURRENT_MATCH(A, B)：
    B 严格匹配 A.current_box；
    // A 上轮已是小框时，下轮仍优先和小框比较

ANCHOR_MATCH(A, B)：
    B 严格匹配 A.anchor_box；
    // B 是可靠完整框，或原来局部可见的 A 恢复完整可见

LOCAL_MATCH(A, B)：
    B 比 A.anchor_box 明显更小，且基本被 anchor_box 包含；
    // 这只是“疑似局部框”，不能立即绑定
```

> 方向不能反过来：`B` 比 `anchor_box` 大、并且把它包含进去时，不能认作同一物品。
>
> 同类候选不唯一时，三个匹配都不成立。


# 3. 对比前

```text
已经得到 N 帧连续无手的稳定 Snapshot；
operation_pending == true；

planned = InventoryDB 的副本；
initial_view[item_id] = 本轮开始前的 view；
binding[snapshot_id] = -1；
matched[item_id] = false；
role[snapshot_id] = NONE；
```

> `role` 只有：`DIRECT / MOVED / REVEALED / RETURNED / NEW_ITEM`。
>
> Snapshot B 只有 `cls_id + box + score`，**没有状态、没有 item_id**。

> 没有 `operation_pending`：只允许 `CURRENT_MATCH` 刷新 `current_box`，不改 `presence` 和 `view`。


# 4. 快照与库存对比

## 1. 先用 current_box 匹配仍看得见的旧物品

```text
for 每个 presence == IN 且 view != HIDDEN 的库存物品 A：

    if 未绑定的 B 唯一满足 CURRENT_MATCH(A, B)：
        A ↔ B；
        planned[A].current_box = B.box；
        planned[A].view = 原来的 view；
        role[B] = DIRECT；
```

> 这一步是核心：已经确认过的局部小框，会一直用 `current_box` 直接匹配。
>
> 所以【局部可见】不必每次重新寻找遮挡物。


## 2. 再用 anchor_box 处理完整可见、露出、放回

```text
for 每个未绑定的 B：

    if B 唯一满足某个 IN + 非 HIDDEN 的 A 的 ANCHOR_MATCH：
        A ↔ B；
        planned[A].current_box = B.box；
        planned[A].view = FULL；
        role[B] = DIRECT；

    else if B 唯一满足某个 IN + HIDDEN 的 A 的 ANCHOR_MATCH：
        A ↔ B；
        planned[A].current_box = B.box；
        planned[A].view = FULL；
        role[B] = REVEALED；

    else if B 唯一满足某个 OUT 的 A 的 ANCHOR_MATCH
            且 return_validation[A, B]：
        A ↔ B；
        planned[A].anchor_box = B.box；
        planned[A].current_box = B.box；
        planned[A].presence = IN；
        planned[A].view = FULL；
        role[B] = RETURNED；
```

> `return_validation` 可以是 ReID、条码或入柜 Track。
>
> 若业务允许“同类同位置即放回”，才把它配置成几何匹配；否则它和“新放入相同物品”无法严格区分。


## 3. Track 只在这里尝试证明【整理】

```text
for 每个未匹配的 IN 物品 A：

    查找 A 的冻结 Track；

    if 未绑定的可靠完整框 B 唯一匹配 Track.release_box：
        A ↔ B；
        planned[A].anchor_box = B.box；
        planned[A].current_box = B.box；
        planned[A].view = FULL；
        role[B] = MOVED；

    else：
        什么都不做；
        // 后续自然退化成 OUT + IN、遮挡或露出
```

> Track **不能覆盖**前两步已经确定的快照绑定。
>
> Track 只保留 item_id，不负责判断放入、取出、遮挡；Track 错了，最多失去【整理】身份，不应拖垮整轮快照结算。
>
> 默认只用可靠 `release_box`；若以后想提高召回率，才把 `Track.path` 作为可选弱证据，不能作为默认结论。


## 4. 再处理“完整框变小”的情况

```text
for 每个尚未匹配的 IN 物品 A：

    if 未绑定的 B 唯一满足 LOCAL_MATCH(A, B)：
        local_candidate[A] = B；
        reserve[B]；
```

> `LOCAL_MATCH` 只说明 B **可能**是 A 剩下的可见部分；也可能是新放进来的同类小物品。

```text
for 每个未绑定、未 reserve 的可靠完整框 C：

    if C 不能匹配任何旧物品、也不能匹配任何 Track 终点：
        maybe_in[C] = true；
```

```text
for 每个 local_candidate[A] = B：

    if 存在 C != B：
        C 是 MOVED 或 maybe_in；
        C 覆盖 A.anchor_box 中 B 没覆盖的区域；

        A ↔ B；
        planned[A].current_box = B.box；
        planned[A].view = PARTIAL；
        role[B] = (initial_view[A] == HIDDEN) ? REVEALED : DIRECT；

    else：
        unresolved[A] = true；
```

> **只有从 FULL / HIDDEN 第一次变成 PARTIAL 时，才需要 C。**
>
> 下一轮 `PARTIAL` 物品已经有 `current_box`，会在第 1 步直接匹配，不再重复证明。


## 5. 没有任何框时：完全遮挡 或 出库

```text
for 每个未匹配、没有 local_candidate、且 view != HIDDEN 的 IN 物品 A：

    if 存在 C：
        C 是 MOVED 或 maybe_in；
        C 覆盖 A.anchor_box 的绝大部分；

        planned[A].current_box = 空；
        planned[A].view = HIDDEN；

    else if A.anchor_box 附近没有任何可信 SnapshotItem：
        planned[A].current_box = 空；
        planned[A].presence = OUT；

    else：
        unresolved[A] = true；
```

> 已经是 `HIDDEN` 的 A，本轮仍没有匹配时保持原样；不能因为继续看不见，就改成 OUT。


## 6. 最后新增物品

```text
for 每个 maybe_in[C]：

    分配新的 item_id；
    anchor_box = C.box；
    current_box = C.box；
    presence = IN；
    view = FULL；
    role[C] = NEW_ITEM；
```

> 新物品第一次入库必须是可靠完整框；只看见局部小框时不新建，因为它没有可靠 `anchor_box`。


# 5. 提交

```text
if 存在 unresolved：
    不提交；

else if 每个可信 SnapshotItem 都有唯一 binding，或已经 NEW_ITEM：
    一次性提交 planned；

else：
    不提交；
```

> 同类物品、同一位置、多种解释都成立时，宁可不提交。
>
> 两个完全相同物品交换位置，或 OUT 物品与新物品完全相同：仅靠 `cls_id + box` 无法严格区分，需要 ReID、条码/RFID 或多视角。
