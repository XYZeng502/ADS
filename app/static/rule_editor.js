/* 规则编辑器 JS —— 独立加载，不影响主页面 */
const ACTION_LABELS = {'SCALE':'放量','GUARD':'控量','STABLE':'维稳','PAUSE':'暂停','BOOST_BUDGET':'加预算','CUT_BUDGET':'减预算','ADJUST_BID':'调整出价'};
const ACTION_COLORS = {'SCALE':'#bbf7d0','GUARD':'#fecaca','STABLE':'#e2e8f0','PAUSE':'#fde68a','BOOST_BUDGET':'#bfdbfe','CUT_BUDGET':'#fecaca','ADJUST_BID':'#fef08a'};
let _cachedRules = [];

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    injectRuleModal();
    var btn = document.getElementById('ruleConfigBtn');
    if (btn) btn.addEventListener('click', openRuleModal);
  });
} else {
  injectRuleModal();
  var btn = document.getElementById('ruleConfigBtn');
  if (btn) btn.addEventListener('click', openRuleModal);
}

function injectRuleModal() {
  if (document.getElementById('ruleModal')) return;
  const html = `
<div id="ruleModal" class="modal-overlay" style="display:none;">
  <div class="modal-dialog modal-lg">
    <div class="modal-header">
      <h2>自定义推荐规则</h2>
      <div style="display:flex;gap:8px;align-items:center;">
        <button class="btn" onclick="showRuleForm()">+ 新建</button>
        <button class="btn" onclick="fetchRules()">刷新</button>
        <button class="modal-close" onclick="closeRuleModal()">&times;</button>
      </div>
    </div>
    <div class="modal-body"><div id="ruleList"></div></div>
  </div>
</div>
<div id="ruleFormModal" class="modal-overlay" style="display:none; z-index:10000;">
  <div class="modal-dialog" style="width:700px;">
    <div class="modal-header">
      <h3 id="ruleFormTitle">新建规则</h3>
      <button class="modal-close" onclick="hideRuleForm()">&times;</button>
    </div>
    <div class="modal-body">
      <input type="hidden" id="ruleEditId" />
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <label>名称<input id="ruleName" placeholder="如：信任高ROI应用"></label>
        <label>匹配应用(*通配)<input id="ruleAppMatch" value="*"></label>
        <label>优先级<input id="rulePriority" type="number" value="100"></label>
        <label>预算调整比例<input id="ruleBudgetRatio" type="number" step="0.01" value="0" placeholder="BOOST/CUT时生效"></label>
        <label>动作<select id="ruleOverrideMode">
          <option value="SCALE">SCALE 放量</option><option value="GUARD">GUARD 控量</option>
          <option value="STABLE">STABLE 维稳</option><option value="PAUSE">PAUSE 暂停</option>
          <option value="BOOST_BUDGET">BOOST_BUDGET 加预算</option>
          <option value="CUT_BUDGET">CUT_BUDGET 减预算</option>
          <option value="ADJUST_BID">ADJUST_BID 调整出价</option>
        </select></label>
      </div>
      <div style="margin-top:10px;"><label>原因<input id="ruleReason" placeholder="如：手动信任白名单" style="width:100%;"></label></div>
      <div style="margin-top:10px;">
        <label style="font-weight:600;">条件（AND）</label>
        <div id="ruleConditions" style="margin-top:4px;"></div>
        <button class="chip" onclick="addConditionRow()" style="margin-top:6px;cursor:pointer;">+ 条件</button>
      </div>
      <div style="margin-top:14px;display:flex;gap:8px;">
        <button class="btn" onclick="saveRule()">保存</button>
        <button class="btn secondary" onclick="hideRuleForm()">取消</button>
      </div>
    </div>
  </div>
</div>
<div id="deleteConfirmModal" class="modal-overlay" style="display:none; z-index:10001;">
  <div class="modal-dialog" style="width:400px;">
    <div class="modal-header"><h3>确认删除</h3></div>
    <div class="modal-body">
      <p id="deleteConfirmMsg" style="margin:0 0 16px;"></p>
      <div style="display:flex;gap:8px;justify-content:flex-end;">
        <button class="btn secondary" onclick="closeDeleteConfirm()">取消</button>
        <button class="btn" style="background:#dc2626;color:#fff;" onclick="executeDeleteRule()">确认删除</button>
      </div>
    </div>
  </div>
</div>`;
  document.body.insertAdjacentHTML('beforeend', html);
}

function openRuleModal() { document.getElementById('ruleModal').style.display = ''; fetchRules(); }
function closeRuleModal() { document.getElementById('ruleModal').style.display = 'none'; }

async function fetchRules() {
  try { const r = await fetch('/web/api/rules'); _cachedRules = (await r.json()).rules || []; renderRuleList(); }
  catch(e) { console.error(e); }
}

function renderRuleList() {
  const el = document.getElementById('ruleList'); if (!el) return;
  if (!_cachedRules.length) { el.innerHTML = '<div style="color:var(--muted);padding:8px 0;">暂无规则。点击「+ 新建」开始。</div>'; return; }
  const rows = typeof appMonthRoiRows === 'function' ? appMonthRoiRows() : [];
  el.innerHTML = _cachedRules.sort((a,b)=>b.priority-a.priority).map(r => {
    const hitCount = rows.filter(row => matchRuleRow(r, row)).length;
    const act = r.action || r.override_mode || 'SCALE';
    const color = ACTION_COLORS[act] || '#f8fafc';
    const badge = `<span class="pill" style="background:${color};">${ACTION_LABELS[act]||act}</span>`;
    const conds = (r.conditions||[]).map(c => c.field+' '+c.op+' '+c.value).join(' & ') || '无条件';
    return `<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid #e2e8f0;${!r.enabled?'opacity:0.5':''}">
      <div style="cursor:pointer;font-size:16px;padding-top:2px;" onclick="toggleRule('${r.id}')">${r.enabled?'🟢':'⚪'}</div>
      <div style="flex:1;min-width:0;">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <b>${r.name}</b>${badge}<span style="color:var(--muted);font-size:11px;">优先${r.priority}</span>
          <span style="color:#059669;font-size:11px;font-weight:600;">命中${hitCount}</span>
        </div>
        <div style="color:var(--muted);font-size:11px;margin-top:2px;">${r.app_match} | ${conds}${r.reason?' | '+r.reason:''}</div>
      </div>
      <div style="display:flex;gap:4px;flex-shrink:0;">
        <button class="chip" onclick="editRuleById('${r.id}')" style="font-size:11px;cursor:pointer;">编辑</button>
        <button class="chip" onclick="confirmDeleteRule('${r.id}')" style="font-size:11px;cursor:pointer;color:#dc2626;">删除</button>
      </div>
    </div>`;
  }).join('');
}

function matchRuleRow(rule, row) {
  if (!rule || !rule.enabled) return false;
  if (rule.app_match !== '*' && rule.app_match !== String(row.app||'')) return false;
  const fieldMap = { month_end_roi_fused:'monthEndRoiFused', month_end_roi_prediction:'monthEndRoi', alert_count:'alertCount', roi_dominant_factor:'dominantFactor', train_days:'trainDays', spend_t1_pred_calibrated:'spendT1Calibrated', actual_spend_last_day:'actualSpend', actual_revenue_last_day:'actualRevenue', d1_anchor_calibrated_mean:'d1Calibrated', month_end_spend_prediction:'monthEndSpend', planned_daily_spend_fused:'plannedDailyFused' };
  for (const c of (rule.conditions||[])) {
    const key = fieldMap[c.field] || c.field;
    const val = row[key]; if (val === undefined || val === null) return false;
    const v = Number(val), cv = Number(c.value);
    if (isNaN(v)||isNaN(cv)) { if ((String(val)===String(c.value)) !== (c.op==='==')) return false; continue; }
    if ((c.op==='>='&&v<cv)||(c.op==='<='&&v>cv)||(c.op==='>'&&v<=cv)||(c.op==='<'&&v>=cv)||(c.op==='=='&&v!==cv)||(c.op==='!='&&v===cv)) return false;
  }
  return true;
}

function editRuleById(ruleId) { const r = _cachedRules.find(x => x.id === ruleId); if (r) showRuleForm(r); }

function showRuleForm(rule) {
  document.getElementById('ruleFormModal').style.display = '';
  document.getElementById('ruleFormTitle').textContent = rule ? '编辑规则' : '新建规则';
  document.getElementById('ruleEditId').value = rule ? rule.id : '';
  document.getElementById('ruleName').value = rule ? (rule.name||'') : '';
  document.getElementById('ruleAppMatch').value = rule ? (rule.app_match||'*') : '*';
  document.getElementById('rulePriority').value = rule ? (rule.priority||100) : 100;
  document.getElementById('ruleOverrideMode').value = rule ? (rule.action||rule.override_mode||'SCALE') : 'SCALE';
  document.getElementById('ruleBudgetRatio').value = rule ? (rule.budget_adjust_ratio||0) : 0;
  document.getElementById('ruleReason').value = rule ? (rule.reason||'') : '';
  document.getElementById('ruleConditions').innerHTML = '';
  (rule ? rule.conditions : []).forEach(c => addConditionRow(c.field, c.op, c.value));
  if (!rule || !rule.conditions || !rule.conditions.length) addConditionRow();
}

function hideRuleForm() { document.getElementById('ruleFormModal').style.display = 'none'; }

function addConditionRow(field, op, value) {
  const fields = [['month_end_roi_fused','月末ROI'],['alert_count','告警数'],['roi_dominant_factor','偏差主因'],['train_days','训练天数'],['spend_t1_pred_calibrated','T+1 Spend'],['actual_spend_last_day','昨日消耗'],['d1_anchor_calibrated_mean','D1校准'],['month_end_spend_prediction','月末消耗']];
  const ops = [['>=','>='],['<=','<='],['>','>'],['<','<'],['==','=='],['!=','!=']];
  const fid = 'c'+Date.now()+Math.random().toString(36).slice(2,6);
  document.getElementById('ruleConditions').insertAdjacentHTML('beforeend',
    `<div style="display:flex;gap:6px;align-items:center;margin-top:6px;padding:8px 10px;background:#f1f5f9;border-radius:8px;border:1px solid #e2e8f0;" id="${fid}">
      <select style="padding:4px 8px;border:1px solid #e2e8f0;border-radius:4px;font-size:13px;background:#fff;">${fields.map(([v,l])=>`<option value="${v}" ${field===v?'selected':''}>${l}</option>`).join('')}</select>
      <select style="padding:4px 8px;border:1px solid #e2e8f0;border-radius:4px;font-size:13px;background:#fff;">${ops.map(([v,l])=>`<option value="${v}" ${op===v?'selected':''}>${l}</option>`).join('')}</select>
      <input value="${value||''}" placeholder="阈值" style="width:100px;padding:4px 8px;border:1px solid #e2e8f0;border-radius:4px;font-size:13px;">
      <button class="chip" onclick="this.closest('div').remove()" style="color:#dc2626;cursor:pointer;">x</button>
    </div>`);
}

async function saveRule() {
  const conds = []; document.querySelectorAll('#ruleConditions > div').forEach(div => {
    const sel = div.querySelectorAll('select'); const inp = div.querySelector('input');
    if (inp && inp.value.trim()) conds.push({field:sel[0].value, op:sel[1].value, value:inp.value.trim()});
  });
  const body = {
    id: document.getElementById('ruleEditId').value || undefined,
    name: document.getElementById('ruleName').value || '未命名',
    app_match: document.getElementById('ruleAppMatch').value || '*',
    priority: parseInt(document.getElementById('rulePriority').value) || 100,
    override_mode: document.getElementById('ruleOverrideMode').value,
    action: document.getElementById('ruleOverrideMode').value,
    budget_adjust_ratio: parseFloat(document.getElementById('ruleBudgetRatio').value) || 0,
    reason: document.getElementById('ruleReason').value || '',
    conditions: conds, enabled: true,
  };
  try {
    const r = await fetch('/web/api/rules', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if ((await r.json()).ok) { hideRuleForm(); fetchRules(); }
  } catch(e) { console.error(e); }
}

function toggleRule(ruleId) {
  const rule = _cachedRules.find(r => r.id === ruleId); if (!rule) return;
  rule.enabled = !rule.enabled;
  fetch('/web/api/rules', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rule)})
    .then(r => r.json()).then(d => { if (d.ok) fetchRules(); });
}

let _pendingDeleteId = null;
function confirmDeleteRule(ruleId) {
  const rule = _cachedRules.find(r => r.id === ruleId);
  document.getElementById('deleteConfirmMsg').textContent = '确定删除规则「'+(rule?rule.name:ruleId)+'」？';
  _pendingDeleteId = ruleId; document.getElementById('deleteConfirmModal').style.display = '';
}
function closeDeleteConfirm() { _pendingDeleteId = null; document.getElementById('deleteConfirmModal').style.display = 'none'; }
async function executeDeleteRule() {
  if (!_pendingDeleteId) return; const rid = _pendingDeleteId; closeDeleteConfirm();
  try { const r = await fetch('/web/api/rules/'+encodeURIComponent(rid), {method:'DELETE'});
    if ((await r.json()).ok) fetchRules(); } catch(e) { console.error(e); }
}
