# pairwise-frame-interpolation

## 领域

游戏、图形与多媒体

## 主题

视频帧插值

## 说明

光流置信、遮挡掩码与场景切换检测。

初始状态下 `media/frame_interpolation.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
