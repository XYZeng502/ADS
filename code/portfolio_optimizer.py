"""
模块6：多产品联合优化模块
使用PuLP库实现线性规划求解
目标：在ROI约束下，最大化全月消耗规模
"""

import pandas as pd
import numpy as np
from pulp import *
from datetime import datetime, timedelta


class PortfolioOptimizer:
    """多产品联合优化类"""
    
    def __init__(self):
        """初始化优化器"""
        self.problem = None
        self.allocation = None

    def optimize_joint_allocation(self, unit_forecasts, monthly_state, kpi_target,
                                  available_budget=None, days_remaining=1,
                                  segment_shares=None, segment_discounts=None):
        """
        产品×投放维度×日期联合优化。

        unit_forecasts 必需列：
        - product_id, date, d1_p20, ltv_multiplier, cap
        可选列：
        - risk_multiplier, d1_p50, d1_p80, 推广流量, 流量场景, 创意规格, 转化类型
        """
        if unit_forecasts is None or len(unit_forecasts) == 0:
            return {
                'allocations': {},
                'unit_allocations': pd.DataFrame(),
                'total_allocated': 0,
                'utilization_rate': 0,
                'status': 'NoData',
                'solver_status': 'NO_UNIT_FORECAST'
            }

        df = unit_forecasts.copy()
        segment_shares = segment_shares or [0.25, 0.25, 0.25, 0.25]
        segment_discounts = segment_discounts or [1.00, 0.95, 0.85, 0.70]

        for col, default in [
            ('risk_multiplier', 1.0),
            ('ltv_multiplier', 1.0),
            ('d1_p20', 0.0),
            ('d1_p50', np.nan),
            ('d1_p80', np.nan),
            ('cap', 0.0),
        ]:
            if col not in df.columns:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

        df['expected_roi_p20'] = df['d1_p20'] * df['ltv_multiplier']
        df['expected_roi_p50'] = df['d1_p50'] * df['ltv_multiplier']
        df['expected_roi_p80'] = df['d1_p80'] * df['ltv_multiplier']
        df['p20_margin_pp'] = (df['expected_roi_p20'] - kpi_target) * 100
        df['p50_margin_pp'] = (df['expected_roi_p50'] - kpi_target) * 100
        df['candidate_rank'] = df['expected_roi_p20'].rank(method='first', ascending=False).astype(int)

        df['effective_cap'] = (
            pd.to_numeric(df['cap'], errors='coerce').fillna(0)
            * pd.to_numeric(df['risk_multiplier'], errors='coerce').fillna(1.0).clip(lower=0, upper=1)
        )
        df = df[df['effective_cap'] > 0].reset_index(drop=True)

        if len(df) == 0:
            return {
                'allocations': {},
                'unit_allocations': pd.DataFrame(),
                'total_allocated': 0,
                'utilization_rate': 0,
                'status': 'NoCap',
                'solver_status': 'NO_EFFECTIVE_CAP'
            }

        candidate_units = self._build_candidate_units(df, kpi_target)

        total_cap = df['effective_cap'].sum()
        budget_limit = float(min(available_budget, total_cap)) if available_budget and available_budget > 0 else float(total_cap)

        actual_spend = float(monthly_state.get('actual_spend', 0))
        actual_revenue = float(monthly_state.get('actual_revenue', 0))
        historical_tail = float(monthly_state.get('historical_tail_revenue', 0))
        baseline_roi = self._baseline_roi(monthly_state)

        self.problem = LpProblem("Joint_Budget_Allocation", LpMaximize)
        variables = []
        records = []

        for idx, row in df.iterrows():
            cap = float(row['effective_cap'])
            for seg_idx, (share, discount) in enumerate(zip(segment_shares, segment_discounts)):
                width = cap * float(share)
                if width <= 0:
                    continue

                var = LpVariable(f"x_{idx}_{seg_idx}", lowBound=0, upBound=width)
                roi_p20 = float(row['d1_p20']) * float(row['ltv_multiplier']) * float(discount)
                variables.append(var)
                objective_score = self._objective_score(row, roi_p20, kpi_target)
                records.append({
                    'var': var,
                    'row_idx': idx,
                    'segment': seg_idx + 1,
                    'segment_width': width,
                    'roi_p20': roi_p20,
                    'roi_p50': (float(row['d1_p50']) * float(row['ltv_multiplier']) * float(discount)
                                if not pd.isna(row.get('d1_p50', np.nan)) else roi_p20),
                    'objective_score': objective_score,
                    'product_id': row.get('product_id'),
                    'date': row.get('date'),
                    'coef': roi_p20 - kpi_target,
                })

        if not variables:
            result = {
                'allocations': {},
                'unit_allocations': pd.DataFrame(),
                'candidate_units': candidate_units,
                'total_allocated': 0,
                'utilization_rate': 0,
                'status': 'NoVariables',
                'solver_status': 'NO_VARIABLES'
            }
            return result

        self.problem += lpSum([rec['objective_score'] * rec['var'] for rec in records]), "Maximize_Quality_Adjusted_Spend"
        self.problem += lpSum(variables) <= budget_limit, "Budget_Limit"

        roi_rhs = kpi_target * actual_spend - actual_revenue - historical_tail
        self.problem += lpSum([rec['coef'] * rec['var'] for rec in records]) >= roi_rhs, "Robust_ROI_Constraint"
        self._add_diversification_constraints(records, budget_limit)

        try:
            self.problem.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
            if self.problem.status != 1:
                print(f"  ⚠️  严格ROI约束不可行，状态码: {self.problem.status}，切换恢复预算模式")
                result = self._create_recovery_joint_allocation(
                    df, monthly_state, kpi_target, budget_limit, days_remaining,
                    segment_shares, segment_discounts, solver_status='RECOVERY_INFEASIBLE'
                )
                result['candidate_units'] = candidate_units
                return result

            segment_rows = []
            for rec in records:
                allocated = rec['var'].varValue or 0
                if allocated <= 1e-6:
                    continue
                source = df.loc[rec['row_idx']].to_dict()
                source.update({
                    'allocated_budget': float(allocated),
                    'segment': rec['segment'],
                    'segment_roi_p20': rec['roi_p20'],
                    'segment_roi_p50': rec['roi_p50'],
                    'objective_score': rec['objective_score'],
                })
                segment_rows.append(source)

            result = self._package_joint_solution(
                segment_rows, monthly_state, kpi_target, budget_limit,
                days_remaining, status='Optimal', solver_status='SUCCESS'
            )
            result['candidate_units'] = candidate_units
            return result

        except Exception as e:
            print(f"  ⚠️  联合优化器出错: {str(e)}，切换恢复预算模式")
            result = self._create_recovery_joint_allocation(
                df, monthly_state, kpi_target, budget_limit, days_remaining,
                segment_shares, segment_discounts, solver_status='RECOVERY_ERROR'
            )
            result['candidate_units'] = candidate_units
            return result

    def _baseline_roi(self, monthly_state):
        actual_spend = float(monthly_state.get('actual_spend', 0))
        actual_revenue = float(monthly_state.get('actual_revenue', 0))
        historical_tail = float(monthly_state.get('historical_tail_revenue', 0))
        if actual_spend <= 0:
            return float(monthly_state.get('actual_roi', 0))
        return (actual_revenue + historical_tail) / actual_spend

    def _objective_score(self, row, roi_p20, kpi_target):
        """主目标仍是规模；小权重奖励高质量、低风险、节假日窗口。"""
        quality_margin = np.clip(roi_p20 - kpi_target, -1.0, 1.0)
        risk = float(row.get('risk_multiplier', 1.0))
        holiday = float(row.get('holiday_scale', 1.0))
        history = float(row.get('historical_weight', 0.0))
        return float(1.0 + 0.02 * quality_margin + 0.002 * risk + 0.001 * (holiday - 1.0) + 0.001 * history)

    def _add_diversification_constraints(self, records, budget_limit):
        """避免可行解在同一产品或同一天过度集中。"""
        if budget_limit <= 0 or not records:
            return

        products = {}
        dates = {}
        for rec in records:
            products.setdefault(rec['product_id'], []).append(rec['var'])
            dates.setdefault(pd.to_datetime(rec['date']).date() if rec.get('date') is not None else None, []).append(rec['var'])

        if len(products) >= 3:
            product_cap = budget_limit * 0.55
            for product_id, vars_ in products.items():
                self.problem += lpSum(vars_) <= product_cap, f"Product_Diversity_{product_id}"

        if len(dates) >= 5:
            date_cap = budget_limit * 0.40
            for date, vars_ in dates.items():
                if date is not None:
                    self.problem += lpSum(vars_) <= date_cap, f"Date_Diversity_{date}"

    def _build_candidate_units(self, df, kpi_target):
        """保留候选单元排行，即使最终因ROI硬约束无法释放预算也能解释模型判断。"""
        candidate = df.copy()
        candidate['expected_roi_p20'] = candidate['d1_p20'] * candidate['ltv_multiplier']
        candidate['expected_roi_p50'] = candidate['d1_p50'] * candidate['ltv_multiplier']
        if 'd1_p80' in candidate.columns:
            candidate['expected_roi_p80'] = candidate['d1_p80'] * candidate['ltv_multiplier']
        candidate['p20_margin_pp'] = (candidate['expected_roi_p20'] - kpi_target) * 100
        candidate['p50_margin_pp'] = (candidate['expected_roi_p50'] - kpi_target) * 100
        candidate['candidate_rank'] = candidate['expected_roi_p20'].rank(method='first', ascending=False).astype(int)
        return candidate.sort_values('expected_roi_p20', ascending=False)

    def _create_greedy_joint_allocation(self, df, monthly_state, kpi_target,
                                        budget_limit, days_remaining,
                                        segment_shares, segment_discounts,
                                        solver_status='FALLBACK'):
        """鲁棒贪心兜底：按保守边际ROI从高到低吃预算段，同时保持ROI约束。"""
        actual_spend = float(monthly_state.get('actual_spend', 0))
        actual_revenue = float(monthly_state.get('actual_revenue', 0))
        historical_tail = float(monthly_state.get('historical_tail_revenue', 0))

        buffer = actual_revenue + historical_tail - kpi_target * actual_spend
        remaining_budget = budget_limit

        segments = []
        for idx, row in df.iterrows():
            cap = float(row['effective_cap'])
            for seg_idx, (share, discount) in enumerate(zip(segment_shares, segment_discounts)):
                width = cap * float(share)
                roi_p20 = float(row['d1_p20']) * float(row['ltv_multiplier']) * float(discount)
                roi_p50 = (float(row['d1_p50']) * float(row['ltv_multiplier']) * float(discount)
                           if not pd.isna(row.get('d1_p50', np.nan)) else roi_p20)
                segments.append({
                    'row_idx': idx,
                    'segment': seg_idx + 1,
                    'width': width,
                    'roi_p20': roi_p20,
                    'roi_p50': roi_p50,
                    'coef': roi_p20 - kpi_target,
                })

        segments.sort(key=lambda x: x['roi_p20'], reverse=True)
        segment_rows = []

        for seg in segments:
            if remaining_budget <= 1e-6:
                break

            width = min(seg['width'], remaining_budget)
            coef = seg['coef']

            if coef >= 0:
                allocated = width
            else:
                if buffer <= 0:
                    continue
                allocated = min(width, buffer / abs(coef))

            if allocated <= 1e-6:
                continue

            buffer += coef * allocated
            remaining_budget -= allocated

            source = df.loc[seg['row_idx']].to_dict()
            source.update({
                'allocated_budget': float(allocated),
                'segment': seg['segment'],
                'segment_roi_p20': seg['roi_p20'],
                'segment_roi_p50': seg['roi_p50'],
            })
            segment_rows.append(source)

        status = 'GreedyFeasible' if buffer >= -1e-6 else 'GreedyRecovery'
        return self._package_joint_solution(
            segment_rows, monthly_state, kpi_target, budget_limit,
            days_remaining, status=status, solver_status=solver_status
        )

    def _create_recovery_joint_allocation(self, df, monthly_state, kpi_target,
                                          budget_limit, days_remaining,
                                          segment_shares, segment_discounts,
                                          solver_status='RECOVERY'):
        """
        ROI硬约束不可行时的恢复预算模式。

        目标不是“释放规模”，而是在受限预算内优先投给 P20 边际ROI高于
        当前组合ROI的单元，尽量抬升组合ROI，并给运营留出可执行的修复方向。
        """
        baseline_roi = self._baseline_roi(monthly_state)
        actual_spend = float(monthly_state.get('actual_spend', 0))
        avg_daily_spend = float(monthly_state.get('avg_daily_spend', 0))
        if avg_daily_spend <= 0 and actual_spend > 0:
            avg_daily_spend = actual_spend / max(float(monthly_state.get('days_elapsed', 21)), 1.0)

        recovery_budget_limit = min(
            budget_limit,
            max(avg_daily_spend * 0.60, actual_spend * 0.015, 1000.0)
        )

        min_lift = 0.01
        segments = []
        for idx, row in df.iterrows():
            cap = float(row['effective_cap'])
            for seg_idx, (share, discount) in enumerate(zip(segment_shares, segment_discounts)):
                width = cap * float(share)
                roi_p20 = float(row['d1_p20']) * float(row['ltv_multiplier']) * float(discount)
                roi_p50 = (float(row['d1_p50']) * float(row['ltv_multiplier']) * float(discount)
                           if not pd.isna(row.get('d1_p50', np.nan)) else roi_p20)
                if roi_p20 <= baseline_roi + min_lift:
                    continue
                lift_score = roi_p20 - baseline_roi
                segments.append({
                    'row_idx': idx,
                    'segment': seg_idx + 1,
                    'width': width,
                    'roi_p20': roi_p20,
                    'roi_p50': roi_p50,
                    'lift_score': lift_score,
                    'objective': lift_score + 0.001 * float(row.get('risk_multiplier', 1.0)) +
                                 0.0005 * float(row.get('holiday_scale', 1.0))
                })

        segments.sort(key=lambda x: x['objective'], reverse=True)

        product_budget = {}
        date_budget = {}
        max_product_share = 0.50
        max_date_share = 0.45
        remaining_budget = recovery_budget_limit
        segment_rows = []

        for seg in segments:
            if remaining_budget <= 1e-6:
                break
            source = df.loc[seg['row_idx']]
            product_id = source.get('product_id')
            date_value = pd.to_datetime(source.get('date'), errors='coerce')
            date_key = date_value.date() if not pd.isna(date_value) else None

            product_remaining = recovery_budget_limit * max_product_share - product_budget.get(product_id, 0.0)
            date_remaining = (recovery_budget_limit * max_date_share - date_budget.get(date_key, 0.0)
                              if date_key is not None else recovery_budget_limit)
            allocated = min(seg['width'], remaining_budget, product_remaining, date_remaining)
            if allocated <= 1e-6:
                continue

            remaining_budget -= allocated
            product_budget[product_id] = product_budget.get(product_id, 0.0) + allocated
            if date_key is not None:
                date_budget[date_key] = date_budget.get(date_key, 0.0) + allocated

            row_data = source.to_dict()
            row_data.update({
                'allocated_budget': float(allocated),
                'segment': seg['segment'],
                'segment_roi_p20': seg['roi_p20'],
                'segment_roi_p50': seg['roi_p50'],
                'recovery_lift_score': seg['lift_score'],
                'objective_score': seg['objective'],
            })
            segment_rows.append(row_data)

        result = self._package_joint_solution(
            segment_rows, monthly_state, kpi_target, budget_limit,
            days_remaining, status='RecoveryBudget', solver_status=solver_status
        )
        result['recovery_mode'] = {
            'baseline_roi': baseline_roi,
            'recovery_budget_limit': recovery_budget_limit,
            'eligible_segments': len(segments),
            'allocated_recovery_budget': result.get('total_allocated', 0),
            'mode_note': 'ROI硬约束不可行；该预算用于提升当前组合ROI，不表示已满足KPI释放条件。'
        }
        return result

    def _package_joint_solution(self, segment_rows, monthly_state, kpi_target,
                                budget_limit, days_remaining, status, solver_status):
        unit_df = pd.DataFrame(segment_rows)
        if unit_df.empty:
            return {
                'allocations': {},
                'unit_allocations': unit_df,
                'total_allocated': 0,
                'utilization_rate': 0,
                'status': status,
                'solver_status': solver_status,
                'predicted_roi_p20': monthly_state.get('actual_roi', 0),
                'predicted_roi_p50': monthly_state.get('actual_roi', 0),
            }

        group_cols = ['product_id', 'date', '推广流量', '流量场景', '创意规格', '转化类型']
        available_group_cols = [c for c in group_cols if c in unit_df.columns]
        agg_dict = {
            'allocated_budget': 'sum',
            'segment_roi_p20': 'mean',
            'segment_roi_p50': 'mean',
            'cap': 'max',
            'effective_cap': 'max',
            'risk_multiplier': 'max',
            'holiday_scale': 'max'
        }
        optional_aggs = {
            'd1_p20': 'mean',
            'd1_p50': 'mean',
            'd1_p80': 'mean',
            'ltv_multiplier': 'mean',
            'expected_roi_p20': 'mean',
            'expected_roi_p50': 'mean',
            'expected_roi_p80': 'mean',
            'p20_margin_pp': 'mean',
            'p50_margin_pp': 'mean',
            'candidate_rank': 'min',
            'historical_weight': 'mean',
            'recovery_lift_score': 'mean',
            'objective_score': 'mean'
        }
        for col, method in optional_aggs.items():
            if col in unit_df.columns:
                agg_dict[col] = method

        unit_summary = unit_df.groupby(available_group_cols, as_index=False).agg(agg_dict)
        unit_summary['daily_budget'] = unit_summary['allocated_budget']
        unit_summary['cap_utilization'] = unit_summary['allocated_budget'] / unit_summary['effective_cap'].replace(0, np.nan)

        product_summary = unit_summary.groupby('product_id', as_index=False)['allocated_budget'].sum()
        allocations = {}
        for _, row in product_summary.iterrows():
            product_id = row['product_id']
            if isinstance(product_id, float) and product_id.is_integer():
                product_id = int(product_id)
            allocated = float(row['allocated_budget'])
            allocations[product_id] = {
                'allocated_budget': allocated,
                'daily_budget': allocated / max(days_remaining, 1),
                'grade': '',
                'optimization': '联合优化分配',
                'allocation_method': 'joint_robust_lp'
            }

        actual_spend = float(monthly_state.get('actual_spend', 0))
        actual_revenue = float(monthly_state.get('actual_revenue', 0))
        historical_tail = float(monthly_state.get('historical_tail_revenue', 0))
        total_allocated = float(unit_summary['allocated_budget'].sum())
        future_revenue_p20 = float((unit_summary['allocated_budget'] * unit_summary['segment_roi_p20']).sum())
        future_revenue_p50 = float((unit_summary['allocated_budget'] * unit_summary['segment_roi_p50']).sum())
        denominator = actual_spend + total_allocated
        predicted_roi_p20 = ((actual_revenue + historical_tail + future_revenue_p20) / denominator
                             if denominator > 0 else 0)
        predicted_roi_p50 = ((actual_revenue + historical_tail + future_revenue_p50) / denominator
                             if denominator > 0 else 0)

        return {
            'allocations': allocations,
            'unit_allocations': unit_summary.sort_values('allocated_budget', ascending=False),
            'total_allocated': total_allocated,
            'utilization_rate': total_allocated / budget_limit if budget_limit > 0 else 0,
            'status': status,
            'solver_status': solver_status,
            'predicted_roi_p20': predicted_roi_p20,
            'predicted_roi_p50': predicted_roi_p50,
            'roi_constraint': {
                'target_roi': kpi_target,
                'actual_spend': actual_spend,
                'actual_revenue': actual_revenue,
                'historical_tail_revenue': historical_tail,
                'future_revenue_p20': future_revenue_p20,
                'future_revenue_p50': future_revenue_p50,
                'baseline_roi': self._baseline_roi(monthly_state),
            }
        }
    
    def optimize_allocation(self, product_diagnoses, available_budget, holiday_info,
                          days_remaining, kpi_target, avg_roi_multiplier=1.0,
                          historical_tail_revenue=0, unit_forecasts=None,
                          monthly_state=None):
        """
        优化产品预算分配

        参数：
        - product_diagnoses: 产品诊断结果 {product_id: diagnosis}
        - available_budget: 可用预算金额
        - holiday_info: 节假日信息 DataFrame
        - days_remaining: 剩余天数
        - kpi_target: KPI目标
        - avg_roi_multiplier: 剩余天数内新投入预算的平均ROI倍率
        - historical_tail_revenue: 历史Cohort在剩余天数的长尾回收

        返回：优化后的预算分配方案
        """
        if unit_forecasts is not None and monthly_state is not None:
            return self.optimize_joint_allocation(
                unit_forecasts=unit_forecasts,
                monthly_state=monthly_state,
                kpi_target=kpi_target,
                available_budget=available_budget,
                days_remaining=days_remaining
            )

        if available_budget <= 0 or len(product_diagnoses) == 0:
            allocation = {}
            for product_id in product_diagnoses.keys():
                allocation[product_id] = {
                    'allocated_budget': 0,
                    'daily_budget': 0,
                    'reason': '无可用预算' if available_budget <= 0 else '无活跃产品'
                }
            return allocation

        # 构建优化问题
        self.problem = LpProblem("Budget_Allocation", LpMaximize)

        # 1. 决策变量：每个产品的分配预算
        product_budgets = {}
        for product_id in product_diagnoses.keys():
            product_budgets[product_id] = LpVariable(f"budget_{product_id}", lowBound=0)

        # 2. 目标函数：最大化总预算（在满足ROI约束的前提下）
        self.problem += lpSum([product_budgets[p] for p in product_budgets.keys()]), "Total_Budget"

        # 3. 约束条件

        # 约束3.1：总预算不超过可用预算
        self.problem += lpSum([product_budgets[p] for p in product_budgets.keys()]) <= available_budget, "Budget_Limit"

        # 约束3.2：ROI约束（核心约束——之前未实际添加到求解器）
        # 月末ROI = (历史回收 + 历史长尾 + 新增回收) / (历史消耗 + 新增消耗) >= KPI
        # 线性化为: sum(x_p * (d1_rate_p * multiplier - KPI)) >= KPI * hist_spend - hist_revenue - hist_tail
        total_hist_spend = 0.0
        total_hist_revenue = 0.0
        roi_lhs = 0.0

        for product_id, diagnosis in product_diagnoses.items():
            stats = diagnosis['statistics']
            total_hist_spend += stats['total_spend']
            total_hist_revenue += stats['total_revenue']

            # 新增预算在剩余天数内能贡献的预期ROI
            d1_rate = diagnosis['efficiency_status']['d1_rate']
            expected_new_roi = d1_rate * avg_roi_multiplier

            # 每单位新增预算对ROI约束的净贡献系数
            coef = expected_new_roi - kpi_target
            roi_lhs += coef * product_budgets[product_id]

        # 右侧：需要新增预算弥补的回收缺口
        # 正数 = 有缺口需要填补，负数 = 有盈余可以释放
        roi_rhs = kpi_target * total_hist_spend - total_hist_revenue - historical_tail_revenue
        self.problem += roi_lhs >= roi_rhs, "ROI_Constraint"

        # 约束3.3：产品级约束（基于诊断结果）
        for product_id, diagnosis in product_diagnoses.items():
            grade = diagnosis['efficiency_status']['grade']
            stats = diagnosis['statistics']
            current_spend = stats['total_spend']
            
            if grade == 'A':
                # A级产品：可以增加预算，最高增加50%
                self.problem += product_budgets[product_id] <= current_spend * 0.5, f"{product_id}_A_limit"
            
            elif grade == 'B':
                # B级产品：可以小幅增加或维持
                self.problem += product_budgets[product_id] <= current_spend * 0.2, f"{product_id}_B_limit"
            
            else:  # C级
                # C级产品：基本不分配，除非有ROI盈余
                # 改为：最多只能分配现有消耗的10%
                self.problem += product_budgets[product_id] <= max(current_spend * 0.1, 100), f"{product_id}_C_limit"
        
        # 4. 求解
        try:
            self.problem.solve(PULP_CBC_CMD(msg=0))
            
            if self.problem.status != 1:  # 1 = Optimal
                print(f"  ⚠️  优化未找到最优解，状态码: {self.problem.status}")
                return self._create_heuristic_allocation(product_diagnoses, available_budget)
            
            # 5. 提取结果
            allocation = {}
            for product_id in product_diagnoses.keys():
                allocated = product_budgets[product_id].varValue
                if allocated is None:
                    allocated = 0
                
                allocation[product_id] = {
                    'allocated_budget': max(allocated, 0),
                    'daily_budget': max(allocated, 0) / max(days_remaining, 1),
                    'grade': product_diagnoses[product_id]['efficiency_status']['grade'],
                    'optimization': product_diagnoses[product_id]['optimization']['direction']
                }
            
            total_allocated = sum([a['allocated_budget'] for a in allocation.values()])
            
            return {
                'allocations': allocation,
                'total_allocated': total_allocated,
                'utilization_rate': total_allocated / available_budget if available_budget > 0 else 0,
                'status': 'Optimal',
                'solver_status': 'SUCCESS'
            }
        
        except Exception as e:
            print(f"  ⚠️  优化器出错: {str(e)}，使用启发式分配")
            return self._create_heuristic_allocation(product_diagnoses, available_budget)
    
    def _create_heuristic_allocation(self, product_diagnoses, available_budget):
        """
        启发式分配（当求解器失败时使用）
        
        逻辑：按产品效率等级进行分配
        - A级：50%
        - B级：35%
        - C级：15%
        """
        allocation = {}
        
        # 统计各级产品
        a_products = []
        b_products = []
        c_products = []
        
        for product_id, diagnosis in product_diagnoses.items():
            grade = diagnosis['efficiency_status']['grade']
            current_spend = diagnosis['statistics']['total_spend']
            
            if grade == 'A':
                a_products.append((product_id, current_spend))
            elif grade == 'B':
                b_products.append((product_id, current_spend))
            else:
                c_products.append((product_id, current_spend))
        
        # 按权重分配
        a_budget = available_budget * 0.5
        b_budget = available_budget * 0.35
        c_budget = available_budget * 0.15
        
        # 在各级内部按消耗规模比例分配
        for product_id, current_spend in a_products + b_products + c_products:
            if len(a_products) > 0 and (product_id, current_spend) in a_products:
                total_a_spend = sum([s for _, s in a_products])
                share = current_spend / total_a_spend if total_a_spend > 0 else 0
                allocated = a_budget * share
            elif len(b_products) > 0 and (product_id, current_spend) in b_products:
                total_b_spend = sum([s for _, s in b_products])
                share = current_spend / total_b_spend if total_b_spend > 0 else 0
                allocated = b_budget * share
            else:
                total_c_spend = sum([s for _, s in c_products])
                share = current_spend / total_c_spend if total_c_spend > 0 else 0
                allocated = c_budget * share
            
            allocation[product_id] = {
                'allocated_budget': max(allocated, 0),
                'daily_budget': max(allocated, 0) / 1,  # 简化为日均
                'grade': product_diagnoses[product_id]['efficiency_status']['grade'],
                'optimization': product_diagnoses[product_id]['optimization']['direction'],
                'allocation_method': 'heuristic'
            }
        
        total_allocated = sum([a['allocated_budget'] for a in allocation.values()])
        
        return {
            'allocations': allocation,
            'total_allocated': total_allocated,
            'utilization_rate': total_allocated / available_budget if available_budget > 0 else 0,
            'status': 'Heuristic',
            'solver_status': 'FALLBACK'
        }
    
    def allocate_by_time_window(self, allocation, holiday_info, days_remaining):
        """
        按时间窗口分配预算
        
        参数：
        - allocation: 优化后的产品级预算分配
        - holiday_info: 节假日scale因子信息
        - days_remaining: 剩余天数
        
        返回：按日期细分的分配方案
        """
        # 选取scale_factor最高的日期优先投放
        holiday_info_sorted = holiday_info.sort_values('scale_factor', ascending=False)
        
        time_allocation = {}
        
        for product_id, allocation_info in allocation.get('allocations', {}).items():
            total_budget = allocation_info['allocated_budget']
            
            # 分散到不同日期
            daily_budgets = {}
            for idx, row in holiday_info_sorted.iterrows():
                date = row['date']
                scale = row['scale_factor']
                
                # 按scale加权分配
                daily_budgets[date] = total_budget * (scale / holiday_info_sorted['scale_factor'].sum())
            
            time_allocation[product_id] = daily_budgets
        
        return time_allocation
