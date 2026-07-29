我们对库存物品 A 增加一个 .first_box 属性, 在物品 A 第一次入库时 A.first_box=A.box,
这个first_box并非是临时的(一轮的),而是一直存在的, 而且只在第一次入库时写入, 后续都不修改


并且修改 block_ids：
    以前的 block_ids 是一个 set(), 直接 A.block_ids.add(B.item_id)
    我希望现在是一个 dict(), 这样使用 A.block_ids[B.item_id] = A.box
即：
    我希望记录遮挡物品A的物品B的同时, 记录他们遮挡时A当时被遮挡前的box是怎么样的
    这样或许当物品B被移走后, 或许能知道A的box变成怎么样？
但是：
    如果A被多个不同的物品遮挡了, 然后某个物品被移走, 但是使用这时的A.block_ids[B.item_id] 得到的A.box也没有用啊
    还不如直接用 A.first_box 来完全进行计算(即：计算A是否完全被遮挡的话, 就用A.first_box减去所有的A.block_ids中的.first_box不就行了)
所以：
    在有了 A.first_box 之后，其实没必要修改 block_ids 了吧？



A.first.box 包含 B.box：
    A.first.x1 <= B.box.x1 + contain_eps
    A.first.y1 <= B.box.y1 + contain_eps
    A.first.x2 >= B.box.x2 - contain_eps
    A.first.y2 >= B.box.y2 - contain_eps


其实我有点大胆的想法就是:
    在证据不足时，优先解释为“旧的遮挡 A 重新露出”，
    而不是“新同类 B 入库并重新挡住 A”。

    即便 A 是【遮挡】物品, 只要最后依旧有:
    !A.affirm, 并且 B.item_id == -1, 并且 IoM(A.box,B.box)≈1, 并且 A.first.box包含B.box, 并且在不把B当作遮挡A的情况下A没有被完全遮挡
    那么我们就大胆认为 A 从【遮挡】变成【可见】, A与B匹配
    否者我们就认为A还是【遮挡】物品, B是新进入的物品, B把A挡住了