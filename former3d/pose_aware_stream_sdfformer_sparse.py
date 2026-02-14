        else:
            # 重复历史特征以填满当前数量（使用torch.repeat确保维度对齐）
            repeat_times = (num_current + num_historical - 1) // num_historical
            projected_features = historical_features.repeat(repeat_times, 1)[:num_current]
            projected_sdfs = torch.zeros(num_current, 1, device=device)
            # 计算需要重复的完整次数和剩余数量
            repeat_count = (num_current + num_historical - 1) // num_historical  # 向上取整
            # 拼接重复的历史特征
            projected_features_list = [historical_features] * repeat_count
            projected_features = torch.cat(projected_features_list, dim=0)[:num_current]
            projected_sdfs = torch.zeros(num_current, 1, device=device)

