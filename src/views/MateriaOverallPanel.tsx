import * as React from 'react';
import * as mobx from 'mobx';
import * as mobxReact from 'mobx-react-lite';
import classNames from 'clsx';
import { Button } from '@rmwc/button';
import { Tab, TabBar } from '@rmwc/tabs';
import { Switch } from '@rmwc/switch';
import { Badge } from '@rmwc/badge';
import { TextField } from '@rmwc/textfield';
import * as G from '../game';
import { useStore } from './components/contexts';

export const MateriaOverallPanel = mobxReact.observer(() => {
  const store = useStore();
  const materiaDetDhtOptimizationAvailable = !store.isViewing && store.schema.mainStat !== undefined;
  const materiaGcdOptimizationAvailable = !store.isViewing &&
    (store.schema.stats.includes('SKS') || store.schema.stats.includes('SPS'));
  let activeTab = store.materiaOverallActiveTab;
  if ((activeTab === 1 && !materiaDetDhtOptimizationAvailable) ||
    (activeTab === 2 && !materiaGcdOptimizationAvailable)) {
    activeTab = 0;
    if (store.materiaOverallActiveTab !== 0) {
      store.setMateriaOverallActiveTab(0);
    }
  }
  return (
    <div className="materia-overall card">
      <div className="materia-overall_tabbar">
        <TabBar
          activeTabIndex={activeTab}
          onActivate={e => {
            store.setMateriaOverallActiveTab(e.detail.index);
            if (activeTab === 1) {
              store.promotion.off('materiaDetDhtOptimization');
            }
          }}
        >
          <Tab>用量预估</Tab>
          {materiaDetDhtOptimizationAvailable && (
            <Tab>
              信念/直击分配优化
              <Badge className="badge-button_badge" exited={!store.promotion.get('materiaDetDhtOptimization')} />
            </Tab>
          )}
          {materiaGcdOptimizationAvailable && (
            <Tab>GCD自动镶嵌</Tab>
          )}
        </TabBar>
      </div>
      {activeTab === 0 && (
        <table className="materia-consumption table">
          <thead>
          <tr>
            <th>魔晶石</th>
            <th>安全孔</th>
            <th>期望</th>
            <th>90%*</th>
            <th>99%*</th>
          </tr>
          </thead>
          <tbody>
          {(() => {
            const ret = [];
            const stats = store.schema.stats.filter(stat => stat in store.materiaConsumption);
            for (const stat of stats) {
              for (const grade of G.materiaGrades) {
                const consumptionItem = store.materiaConsumption[stat]![grade];
                if (consumptionItem === undefined) continue;
                ret.push((
                  <tr key={stat + grade}>
                    <td>{G.getMateriaName(stat, grade, store.setting.materiaDisplayName === 'stat')}</td>
                    <td>{consumptionItem.safe}</td>
                    <td>{consumptionItem.expectation}</td>
                    <td>{consumptionItem.confidence90}</td>
                    <td>{consumptionItem.confidence99}</td>
                  </tr>
                ));
              }
            }
            return ret;
          })()}
          {Object.keys(store.materiaConsumption).length === 0 && (
            <tr className="materia-consumption_empty">
              <td colSpan={5}>未镶嵌魔晶石</td>
            </tr>
          )}
          <tr className="materia-consumption_tip">
            <td colSpan={5}>
              *以此总体成功率完成全部镶嵌所需的数量
              {store.schema.toolMateriaDuplicates! > 1 && (
                <div
                  className={classNames(
                    'materia-consumption_tool-duplicates',
                    !store.duplicateToolMateria && '-disabled',
                  )}
                >
                  {`主副手的用量按照${store.schema.toolMateriaDuplicates}套计算`}
                  <Switch
                    className="materia-consumption_tool-duplicates-switch"
                    checked={store.duplicateToolMateria}
                    onChange={store.toggleDuplicateToolMateria}
                  />
                </div>
              )}
            </td>
          </tr>
          </tbody>
        </table>
      )}
      {activeTab === 1 && (
        <MateriaDetDhtOptimization />
      )}
      {activeTab === 2 && materiaGcdOptimizationAvailable && (
        <MateriaGcdOptimization />
      )}
    </div>
  );
});

const MateriaDetDhtOptimization = mobxReact.observer(() => {
  const store = useStore();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const solutions = React.useMemo(() => mobx.untracked(() => store.materiaDetDhtOptimized), []);
  return (
    <div className="materia-det-dht-optimization">
      <div className="materia-det-dht-optimization_introduce">
        <p>根据增伤期望优化配装的信念、直击魔晶石分配。</p>
        <p>{'直击属性对必定直击型技能的固定增伤量与对一般技能期望增伤量存在小幅差距，' +
            '直击类团辅也会小幅降低直击属性的收益，这些因素未被纳入考虑。'}</p>
        <p>{'选中的装备中，已镶嵌信念或直击魔晶石的孔洞和空置的孔洞将被视为可使用孔洞，' +
            '每个可使用孔洞将会被镶嵌此孔洞可镶嵌的最高等级魔晶石。'}</p>
      </div>
      <table className="materia-det-dht-optimization_solutions table">
        <thead>
        <tr>
          <th>信念</th>
          <th>直击</th>
          <th style={{ width: '99%' }} />
        </tr>
        </thead>
        <tbody>
        {solutions.map((solution, i) => (
          <tr key={i}>
            <td>{solution.DET}</td>
            <td>{solution.DHT}</td>
            <td>
              {(solution.DET === store.equippedStats['DET'] && solution.DHT === store.equippedStats['DHT']) ? (
                '已使用此方案'
              ) : (
                <Button
                  className="materia-det-dht-optimization_use-solution"
                  onClick={() => store.setMateriaDetDhtOptimization(solution.gearMateriaStats)}
                  children="使用此方案"
                />
              )}
            </td>
          </tr>
        ))}
        </tbody>
      </table>
      <div className="materia-det-dht-optimization_tip">
        方案按增伤期望从高到低排序，所有方案的差距小于万分之三。
      </div>
    </div>
  );
});

const MateriaGcdOptimization = mobxReact.observer(() => {
  const store = useStore();
  const speedStat = (store.schema.stats.includes('SKS') && 'SKS') ||
    (store.schema.stats.includes('SPS') && 'SPS');
  const equippedEffects = store.equippedEffects;
  const [ targetGcd, setTargetGcd ] = React.useState<string>(() =>
    (equippedEffects?.gcd ?? 2.5).toFixed(2));
  const [ status, setStatus ] = React.useState<string>('');
  const equippedAllGears = store.schema.slots.every(slot => store.equippedGears.get(slot.slot.toString()) !== undefined);
  const equippedFood = store.equippedGears.get('-1') !== undefined;
  const ready = equippedAllGears && equippedFood;

  return (
    <div className="materia-gcd-optimization">
      <div className="materia-gcd-optimization_form">
        <TextField
          className="materia-gcd-optimization_input"
          label="目标GCD（秒）"
          type="number"
          step={0.01}
          value={targetGcd}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTargetGcd(e.target.value)}
        />
        <Button
          className="materia-gcd-optimization_button"
          raised
          disabled={!ready || speedStat === undefined}
          onClick={() => {
            const result = store.optimizeMateriaForGcd(Number(targetGcd));
            if (!result.success) {
              const achieved = result.achievedGcd !== undefined && !Number.isNaN(result.achievedGcd)
                ? result.achievedGcd.toFixed(2)
                : '--';
              setStatus(`未能完全满足目标GCD，当前方案 GCD 为 ${achieved}，已套用可行的最优方案。` +
                '如需更快GCD，请尝试增加速度目标的宽容度或调整装备/食物。');
            } else {
              setStatus(`已套用方案：GCD ${result.achievedGcd?.toFixed(2)}，期望伤害 ${result.damage?.toFixed(3)}`);
            }
          }}
        >
          计算并套用
        </Button>
      </div>
      <div className="materia-gcd-optimization_hint">
        {ready ? `当前GCD：${equippedEffects?.gcd.toFixed(2) ?? '--'}，` : '请先选齐所有装备并选择食物后再尝试。'}
        仅在保持目标GCD的前提下尝试最大化每威力伤害期望。
      </div>
      {status.length > 0 && (
        <div className="materia-gcd-optimization_status">{status}</div>
      )}
    </div>
  );
});
