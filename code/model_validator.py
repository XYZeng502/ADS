"""
模型验证与离线对比报告。

验证目标：
1. 用时间切分避免数据泄漏，验证 D1 分位数预测是否优于产品均值基线。
2. 在验证集真实活跃投放单元上，用同等预算比较新算法排序与旧产品均值排序。
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from holiday_identifier import HolidayIdentifier
from ltv_predictor import LTVPredictor
from ml_predictor import MLPredictor, PLACEMENT_DIMS


class ModelValidator:
    """时间切分验证器。"""

    def __init__(self, data, kpi_target=1.05, validation_days=7, holiday_identifier=None):
        self.data = data.copy()
        self.data['日期'] = pd.to_datetime(self.data['日期'])
        self.kpi_target = kpi_target
        self.validation_days = validation_days
        self.holiday_identifier = holiday_identifier or HolidayIdentifier()

    def run(self, output_dir, timestamp=None):
        """执行验证并输出 Markdown + CSV 明细。"""
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)

        train_data, validation_data, split_info = self._split_by_time()

        ltv = LTVPredictor(train_data)
        ltv.calculate_ltv_curves()

        predictor = MLPredictor(train_data, self.holiday_identifier)
        ml_trained = predictor.train()

        prediction_detail = self._evaluate_predictions(
            predictor, train_data, validation_data
        )
        prediction_metrics = self._prediction_metrics(prediction_detail)

        allocation_detail = self._evaluate_allocation_ranking(
            predictor, ltv, train_data, validation_data
        )
        allocation_metrics = self._allocation_metrics(allocation_detail)

        prediction_path = os.path.join(output_dir, f"validation_prediction_detail_{timestamp}.csv")
        allocation_path = os.path.join(output_dir, f"validation_allocation_detail_{timestamp}.csv")
        report_path = os.path.join(output_dir, f"validation_report_{timestamp}.md")

        prediction_detail.to_csv(prediction_path, index=False)
        allocation_detail.to_csv(allocation_path, index=False)

        markdown = self._build_markdown_report(
            split_info=split_info,
            ml_trained=ml_trained,
            prediction_metrics=prediction_metrics,
            allocation_metrics=allocation_metrics,
            prediction_path=prediction_path,
            allocation_path=allocation_path
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        return {
            'report_path': report_path,
            'prediction_detail_path': prediction_path,
            'allocation_detail_path': allocation_path,
            'prediction_metrics': prediction_metrics,
            'allocation_metrics': allocation_metrics,
            'split_info': split_info,
            'ml_trained': ml_trained
        }

    def _split_by_time(self):
        dates = sorted(self.data['日期'].dropna().unique())
        if len(dates) < 10:
            validation_count = max(1, len(dates) // 4)
        else:
            validation_count = min(self.validation_days, max(1, len(dates) // 4))

        validation_dates = dates[-validation_count:]
        train_end = validation_dates[0] - pd.Timedelta(days=1)
        train_data = self.data[self.data['日期'] < validation_dates[0]].copy()
        validation_data = self.data[self.data['日期'].isin(validation_dates)].copy()

        split_info = {
            'train_start': train_data['日期'].min(),
            'train_end': train_end,
            'validation_start': validation_data['日期'].min(),
            'validation_end': validation_data['日期'].max(),
            'train_rows': len(train_data),
            'validation_rows': len(validation_data),
            'train_days': train_data['日期'].nunique(),
            'validation_days': validation_data['日期'].nunique(),
            'train_products': train_data['应用ID'].nunique(),
            'validation_products': validation_data['应用ID'].nunique(),
        }
        return train_data, validation_data, split_info

    def _evaluate_predictions(self, predictor, train_data, validation_data):
        rows = []
        all_data = pd.concat([train_data, validation_data], ignore_index=True).sort_values('日期')

        for _, row in validation_data.iterrows():
            spend = row.get('消耗金额', 0)
            d1_revenue = row.get('首日广告收入', 0)
            if pd.isna(spend) or spend <= 0 or pd.isna(d1_revenue):
                continue

            date = pd.to_datetime(row['日期'])
            pid = row['应用ID']
            history = all_data[all_data['日期'] < date]
            recent_context = self._recent_context(pid, date, history, predictor)

            q = predictor.predict_d1_quantiles(pid, date, placement_row=row, recent_context=recent_context)
            baseline = self._baseline_d1(pid, predictor)
            actual = d1_revenue / spend

            rows.append({
                '日期': date.strftime('%Y-%m-%d'),
                '产品ID': pid,
                '推广流量': row.get('推广流量', 0),
                '流量场景': row.get('流量场景', 0),
                '创意规格': row.get('创意规格', 0),
                '转化类型': row.get('转化类型', 0),
                '消耗': spend,
                '实际D1': actual,
                '模型P20': q['p20'],
                '模型P50': q['p50'],
                '模型P80': q['p80'],
                '产品均值基线': baseline,
                '模型绝对误差': abs(q['p50'] - actual),
                '基线绝对误差': abs(baseline - actual),
                'P20命中': int(actual >= q['p20']),
                'P80命中': int(actual <= q['p80']),
                'P20_P80区间命中': int(q['p20'] <= actual <= q['p80']),
                'P20保守余量': actual - q['p20'],
                'P50误差': q['p50'] - actual,
            })

        return pd.DataFrame(rows)

    def _evaluate_allocation_ranking(self, predictor, ltv, train_data, validation_data):
        rows = []
        all_data = pd.concat([train_data, validation_data], ignore_index=True).sort_values('日期')

        for date in sorted(validation_data['日期'].unique()):
            date = pd.to_datetime(date)
            history = all_data[all_data['日期'] < date].copy()
            actual_day = validation_data[validation_data['日期'] == date].copy()
            if history.empty or actual_day.empty:
                continue

            product_ids = list(history['应用ID'].unique())
            holiday_info = self.holiday_identifier.get_holiday_scale_factors(date, date)
            forecasts = predictor.predict_remaining_days_quantiles(
                product_ids=product_ids,
                start_date=date,
                end_date=date,
                historical_data=history,
                holiday_info_df=holiday_info
            )
            if forecasts.empty:
                continue

            actual_units = self._actual_units(actual_day)
            candidates = forecasts.merge(
                actual_units,
                on=['product_id'] + PLACEMENT_DIMS,
                how='inner'
            )
            if candidates.empty:
                continue

            month_end = self._month_end(date)
            candidates['ltv_multiplier'] = candidates.apply(
                lambda r: ltv.get_multiplier(
                    r['product_id'],
                    max(1, (month_end - pd.to_datetime(r['date'])).days + 1),
                    unit_key=tuple([r['product_id']] + [r.get(dim, 0) for dim in PLACEMENT_DIMS])
                ),
                axis=1
            )
            candidates['新算法排序分'] = candidates['d1_p20'] * candidates['ltv_multiplier']
            candidates['基线排序分'] = candidates.apply(
                lambda r: self._baseline_d1(r['product_id'], predictor) * r['ltv_multiplier'],
                axis=1
            )
            candidates['可验证Cap'] = np.minimum(
                candidates['cap'].fillna(0),
                candidates['实际消耗'].fillna(0)
            )
            candidates = candidates[candidates['可验证Cap'] > 0].copy()
            if candidates.empty:
                continue

            # 同等预算验证：取当日实际消耗的25%，且不超过可观测cap的50%。
            budget = min(
                actual_day['消耗金额'].sum() * 0.25,
                candidates['可验证Cap'].sum() * 0.50
            )
            if budget <= 0:
                continue

            new_rows = self._greedy_select(candidates, '新算法排序分', budget, '新算法P20排序')
            baseline_rows = self._greedy_select(candidates, '基线排序分', budget, '产品均值基线')
            rows.extend(new_rows)
            rows.extend(baseline_rows)

        return pd.DataFrame(rows)

    def _actual_units(self, actual_day):
        group_cols = ['应用ID'] + [d for d in PLACEMENT_DIMS if d in actual_day.columns]
        grouped = actual_day.groupby(group_cols, as_index=False).agg({
            '消耗金额': 'sum',
            '首日广告收入': 'sum'
        })
        grouped = grouped.rename(columns={
            '应用ID': 'product_id',
            '消耗金额': '实际消耗',
            '首日广告收入': '实际D1收入'
        })
        grouped['实际D1'] = grouped['实际D1收入'] / grouped['实际消耗'].replace(0, np.nan)
        return grouped.dropna(subset=['实际D1'])

    def _greedy_select(self, candidates, score_col, budget, strategy):
        selected = []
        remaining = budget
        ranked = candidates.sort_values(score_col, ascending=False)

        for _, row in ranked.iterrows():
            if remaining <= 1e-6:
                break
            allocated = min(float(row['可验证Cap']), remaining)
            if allocated <= 0:
                continue
            remaining -= allocated
            selected.append({
                '日期': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                '策略': strategy,
                '产品ID': row['product_id'],
                '推广流量': row.get('推广流量', 0),
                '流量场景': row.get('流量场景', 0),
                '创意规格': row.get('创意规格', 0),
                '转化类型': row.get('转化类型', 0),
                '分配预算': allocated,
                '排序分': row[score_col],
                '实际D1': row['实际D1'],
                '月内LTV倍率': row['ltv_multiplier'],
                '实际月内ROI代理': row['实际D1'] * row['ltv_multiplier'],
                '模型D1_P20': row.get('d1_p20', np.nan),
                '模型D1_P50': row.get('d1_p50', np.nan),
                '基线D1': self._baseline_d1(row['product_id'], None),
            })
        return selected

    def _prediction_metrics(self, detail):
        if detail.empty:
            return {}

        spend = detail['消耗'].clip(lower=0)
        spend_sum = spend.sum()

        def weighted_mean(series):
            return float(np.average(series, weights=spend)) if spend_sum > 0 else float(series.mean())

        baseline_mae = detail['基线绝对误差'].mean()
        model_mae = detail['模型绝对误差'].mean()
        baseline_wmae = weighted_mean(detail['基线绝对误差'])
        model_wmae = weighted_mean(detail['模型绝对误差'])

        return {
            '样本数': int(len(detail)),
            '验证消耗': float(detail['消耗'].sum()),
            '模型MAE': float(model_mae),
            '基线MAE': float(baseline_mae),
            'MAE提升': float((baseline_mae - model_mae) / baseline_mae) if baseline_mae > 0 else np.nan,
            '模型加权MAE': float(model_wmae),
            '基线加权MAE': float(baseline_wmae),
            '加权MAE提升': float((baseline_wmae - model_wmae) / baseline_wmae) if baseline_wmae > 0 else np.nan,
            'P20覆盖率_实际>=P20': float(detail['P20命中'].mean()),
            'P80覆盖率_实际<=P80': float(detail['P80命中'].mean()),
            'P20_P80区间覆盖率': float(detail['P20_P80区间命中'].mean()),
            'P20平均保守余量': float(detail['P20保守余量'].mean()),
        }

    def _allocation_metrics(self, detail):
        if detail.empty:
            return {}

        grouped = detail.groupby('策略').apply(self._allocation_summary, include_groups=False).reset_index()
        result = {'策略汇总': grouped}
        pivot = grouped.set_index('策略')
        if '新算法P20排序' in pivot.index and '产品均值基线' in pivot.index:
            new_roi = pivot.loc['新算法P20排序', '实际月内ROI代理']
            base_roi = pivot.loc['产品均值基线', '实际月内ROI代理']
            new_d1 = pivot.loc['新算法P20排序', '实际D1']
            base_d1 = pivot.loc['产品均值基线', '实际D1']
            result.update({
                '实际月内ROI代理提升pp': float((new_roi - base_roi) * 100),
                '实际D1提升pp': float((new_d1 - base_d1) * 100),
                '预算对比口径': '同等预算，且只在验证日真实活跃单元内比较'
            })
        return result

    def _allocation_summary(self, grp):
        weights = grp['分配预算'].clip(lower=0)
        weight_sum = weights.sum()

        def weighted(col):
            return float(np.average(grp[col], weights=weights)) if weight_sum > 0 else float(grp[col].mean())

        return pd.Series({
            '分配预算': float(weight_sum),
            '选择单元数': int(len(grp)),
            '实际D1': weighted('实际D1'),
            '实际月内ROI代理': weighted('实际月内ROI代理'),
            '平均排序分': weighted('排序分'),
            '平均LTV倍率': weighted('月内LTV倍率'),
        })

    def _recent_context(self, pid, date, history, predictor):
        prod_hist = history[history['应用ID'] == pid]
        recent_3d = prod_hist[prod_hist['日期'] >= date - pd.Timedelta(days=3)]
        recent_7d = prod_hist[prod_hist['日期'] >= date - pd.Timedelta(days=7)]
        baseline = self._baseline_d1(pid, predictor)

        def safe_d1(subset):
            spend = subset['消耗金额'].sum()
            return subset['首日广告收入'].sum() / spend if spend > 0 else baseline

        return {
            'recent_3d_d1': safe_d1(recent_3d) if len(recent_3d) else baseline,
            'recent_7d_d1': safe_d1(recent_7d) if len(recent_7d) else baseline,
            'recent_3d_spend': recent_3d['消耗金额'].mean() if len(recent_3d) else 0,
            'recent_7d_spend': recent_7d['消耗金额'].mean() if len(recent_7d) else 0,
        }

    def _baseline_d1(self, pid, predictor):
        if predictor is not None and pid in predictor.product_medians:
            return float(predictor.product_medians[pid].get('median_d1', 0.65))
        product_data = self.data[self.data['应用ID'] == pid]
        spend = product_data['消耗金额'].sum()
        return float(product_data['首日广告收入'].sum() / spend) if spend > 0 else 0.65

    def _month_end(self, date):
        date = pd.to_datetime(date)
        if date.month == 12:
            return pd.to_datetime(f"{date.year + 1}-01-01") - pd.Timedelta(days=1)
        return pd.to_datetime(f"{date.year}-{date.month + 1:02d}-01") - pd.Timedelta(days=1)

    def _format_metric_table(self, metrics):
        rows = []
        for key, value in metrics.items():
            if isinstance(value, pd.DataFrame):
                continue
            if isinstance(value, float):
                rows.append((key, f"{value:.4f}"))
            else:
                rows.append((key, value))
        return self._markdown_table(['指标', '数值'], rows)

    def _markdown_table(self, headers, rows):
        lines = [
            '| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'
        ]
        for row in rows:
            lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
        return '\n'.join(lines)

    def _build_markdown_report(self, split_info, ml_trained, prediction_metrics,
                               allocation_metrics, prediction_path, allocation_path):
        lines = []
        lines.append('# 智能预算决策系统 - 算法验证报告')
        lines.append('')
        lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append('')
        lines.append('## 1. 数据切分')
        split_rows = []
        for key, value in split_info.items():
            if isinstance(value, pd.Timestamp):
                value = value.strftime('%Y-%m-%d')
            split_rows.append((key, value))
        lines.append(self._markdown_table(['字段', '值'], split_rows))
        lines.append('')
        lines.append(f"LightGBM分位数模型训练状态：{'已训练' if ml_trained else '使用层级分位数兜底'}")
        lines.append('')
        lines.append('## 2. D1预测验证')
        lines.append(self._format_metric_table(prediction_metrics))
        lines.append('')
        if prediction_metrics:
            lift = prediction_metrics.get('加权MAE提升', np.nan)
            lines.append(f"结论：相对产品均值基线，模型加权MAE提升 `{lift:.2%}`。")
            lines.append('P20覆盖率越接近80%，说明保守约束越稳定；P20-P80区间覆盖率用于观察不确定性区间是否有解释力。')
        lines.append('')
        lines.append('## 3. 同等预算离线排序验证')
        strategy_df = allocation_metrics.get('策略汇总') if allocation_metrics else None
        if isinstance(strategy_df, pd.DataFrame) and not strategy_df.empty:
            rows = []
            for _, row in strategy_df.iterrows():
                rows.append([
                    row['策略'],
                    f"{row['分配预算']:.2f}",
                    int(row['选择单元数']),
                    f"{row['实际D1']:.2%}",
                    f"{row['实际月内ROI代理']:.2%}",
                    f"{row['平均排序分']:.2%}",
                    f"{row['平均LTV倍率']:.3f}",
                ])
            lines.append(self._markdown_table(
                ['策略', '分配预算', '选择单元数', '实际D1', '实际月内ROI代理', '平均排序分', '平均LTV倍率'],
                rows
            ))
            lines.append('')
            lines.append(
                f"新算法相对产品均值基线：实际D1提升 "
                f"`{allocation_metrics.get('实际D1提升pp', 0):.2f}pp`，"
                f"实际月内ROI代理提升 `{allocation_metrics.get('实际月内ROI代理提升pp', 0):.2f}pp`。"
            )
            lines.append('')
            lines.append(f"比较口径：{allocation_metrics.get('预算对比口径', '')}。")
        else:
            lines.append('验证集中没有足够可观测的共同投放单元，无法进行同等预算排序验证。')
        lines.append('')
        lines.append('## 4. 明细文件')
        lines.append(f"- D1预测明细：`{prediction_path}`")
        lines.append(f"- 预算排序明细：`{allocation_path}`")
        lines.append('')
        lines.append('## 5. 注意事项')
        lines.append('- 离线预算验证只在验证日真实活跃且有实际D1的投放单元内比较，不能完全替代线上A/B扩量实验。')
        lines.append('- “实际月内ROI代理”使用验证日实际D1乘以训练集LTV倍率，用于近似比较排序质量。')
        lines.append('- 如果P20覆盖率明显偏离80%，应继续校准分位数模型或提高保守折扣。')
        lines.append('')
        return '\n'.join(lines)
