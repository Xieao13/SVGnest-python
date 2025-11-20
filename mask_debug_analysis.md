# Mask更新逻辑Debug分析报告

## 🔍 发现的核心问题

### 1. **Mask和环境状态不同步** ⚠️⚠️⚠️

**现象**：
- 有时mask显示"0 parts already selected"，但环境报告action已经被选择
- 导致模型选择无效动作，获得-1惩罚

**原因**：
- 环境基于`placement_sequence`检查无效动作
- 模型基于`selected_mask`阻止选择已选零件
- 两者在某些情况下不同步

**示例**：
```
Mask status: 0 parts already selected, mask=[]
⚠️ INVALID ACTION DETECTED! Action 0 was already selected.
```

### 2. **Log Prob不一致** ⚠️⚠️

**现象**：
- Rollout阶段：log_prob均值约 -2.5
- 更新阶段：log_prob均值约 -3.6
- 差异均值：约 1.2

**原因**：
- Rollout时：mask逐步更新，可能包含已选零件
- 更新时：`evaluate_actions`总是从空mask开始
- 导致log_prob计算不一致，ratio计算错误

**影响**：
- 65-90%的ratio被裁剪（超出[0.8, 1.2]范围）
- 策略更新不稳定

### 3. **无效动作频繁出现** ⚠️

**统计**：
- 在测试运行中，多次出现无效动作
- 导致-1惩罚，影响训练效果

## 🔧 解决方案

### 方案1：在Buffer中存储Mask状态（推荐）

**优点**：
- 确保rollout和更新阶段的mask状态一致
- 解决log_prob不一致问题

**实现**：
1. 在`RolloutBuffer`中添加`masks`字段
2. 在`add`方法中存储mask状态
3. 在`evaluate_actions`中使用存储的mask

**代码修改**：
```python
# RolloutBuffer.add()
def add(self, state, action, log_prob, reward, done, value, mask):
    # ... 现有代码 ...
    self.masks[self.pos] = mask.clone()

# evaluate_actions调用
new_log_probs, new_values, entropy = self.policy.evaluate_actions(
    states, actions.unsqueeze(1), masks  # 传入mask
)
```

### 方案2：确保Mask和环境状态同步

**实现**：
1. 在episode重置时，验证mask和环境状态一致
2. 在每次action后，验证mask更新正确
3. 如果发现不一致，强制同步

### 方案3：修改evaluate_actions逻辑

**实现**：
1. 在`evaluate_actions`中，从state和actions序列重建mask
2. 但这需要知道actions的完整序列，在PPO中不可行

## 📊 当前状态

### 正常的部分
✅ 梯度范数正常（2-7）
✅ Episode能正常完成
✅ 材料利用率在0.7-0.8之间
✅ 没有NaN或Inf错误

### 需要修复的部分
❌ Mask和环境状态不同步
❌ Log prob不一致
❌ 无效动作频繁出现
❌ Ratio被大量裁剪

## 🎯 建议的修复优先级

1. **高优先级**：修复mask和环境状态同步问题
2. **高优先级**：在buffer中存储mask，解决log_prob不一致
3. **中优先级**：优化价值函数，降低value loss
4. **低优先级**：调整超参数（学习率、clip_range等）

