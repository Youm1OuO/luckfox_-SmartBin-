当一次操作时, 如果我们用多个物体把一个仓库中原本【可见】的物体A给完全遮挡起来后(即这几个物体都必然在这个A附近)
我们是无法区分开是多个物品与这个被完全遮挡物品A的关系的:
> 我们只知道这些无法都在A附近, 这些物品单独与A的框都是有交集的
> 1. 有可能是这几个物体分别都只能部分遮挡A, 但是这几个物品合力"恰好"将物体A给完全遮挡了
> 2. 也有可能是这几个物体分别都只能部分遮挡A, 但是只需要其中某些物品就能将物体A给完全遮挡了, 有些物体没遮挡A
> 3. 也有可能是有些物体只能部分遮挡A, 但有些物品单凭自己就能把物体A给完全遮挡了, 其他物品没遮挡A (这时也有可能是大家都遮挡了, 主要是看)

也就是说：
    如果 block_ids 想要只记录真实遮挡了 A 的物品的 item_id 的话, 就需要准确知道这些物品C遮挡 A 时的先后遮挡顺序
    但是这样很难做到, 很难对物品进行时间上的排序


有个比较简单的方法:
    只要这个物品C与库存物品A的框有交集, 我们就通通把 C.item_id 添加进 block_ids 的最后面
    (虽然目前要求插在最后也没有实际派上用场, 但是有规律必然比无规律的要好, 而且万一以后用到呢)

并且有个小操作：
    我们在修改物体的 block_ids 时(增加或删除), 把这些信息记录到 temp_add_block_ids 与 temp_reduce_block_ids
    (temp_add_block_ids 与 temp_reduce_block_ids 仅本轮使用)
具体操作是：
    最最开始先把库存所有旧物品A的框的大小都存储下来, 供后续使用
    for for A in 库存物品:
        old_box[A.item_id]=A.box
    我们移走物体C时, 有可能会对一些物品D的 block_ids 删除, 那么同时 temp_reduce_block_ids[D.item_id].add(C)
    我们拿来物体C时, 有可能会对一些物品D的 block_ids 增加, 那么同时 temp_add_block_ids[D.item_id].add(C)


然后修改库存之后, 我们最后检查：
```python
for A in 库存物品:
    if temp_reduce_block_ids[A.item_id] is not none：
        # temp_reduce_block_ids[A.item_id] 不为空, 说明之前遮挡了A的物品有被拿走 (A的框可能会表大)
        remaining_reduce = normalize_box(A.box)
        covered_reduce = Fales
        for C in temp_reduce_block_ids[A.item_id]:
            remaining_reduce, b_covered = apply_cover(remaining=remaining_reduce, new_cover=C.box)
            if b_covered:
                covered_reduce = b_covered
        old_box_ = old_box[A.item_id]
        if IoM(remaining_reduce, old_box_)≈1 
        and remaining_reduce.box.area ≥ old_box_.box.area 
        and A.affirm == True 
        and A的状态 ==【遮挡】
        and matched_snapshot_of_item 中存在键 A.item_id:
            A的状态从【遮挡】变成【可见】
            A.box = matched_snapshot_of_item[A.item_id].box

    if temp_reduce_block_ids[A.item_id] is not none：
            

    remaining, b_covered = apply_cover(remaining=remaining, new_cover=B)
```