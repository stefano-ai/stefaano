import { FormEvent, useState } from 'react';
import { DiceRollRequest, DiceRollResult } from '@dnd-tabletop-sim/shared';

interface DiceTrayProps {
  onRoll: (request: DiceRollRequest) => void;
  diceHistory: DiceRollResult[];
  rollerId?: string;
}

export function DiceTray({ onRoll, diceHistory, rollerId }: DiceTrayProps) {
  const [formula, setFormula] = useState('1d20+0');
  const [context, setContext] = useState('');
  const [advantage, setAdvantage] = useState<'normal' | 'advantage' | 'disadvantage'>('normal');

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!formula.trim()) return;

    onRoll({
      formula: formula.trim(),
      context: context.trim() || undefined,
      rollerId,
      advantage: advantage === 'advantage',
      disadvantage: advantage === 'disadvantage',
    });
  };

  return (
    <section className="flex max-h-64 flex-col border-t border-slate-800 bg-slate-950/60 px-4 py-3">
      <form className="flex flex-wrap items-center gap-3" onSubmit={handleSubmit}>
        <input
          className="min-w-[120px] flex-1 rounded-md border border-slate-700 bg-slate-900 p-2 text-sm text-slate-100 focus:border-orange-400 focus:outline-none"
          value={formula}
          onChange={(event) => setFormula(event.target.value)}
          placeholder="e.g. 2d6+3"
        />
        <input
          className="min-w-[160px] flex-1 rounded-md border border-slate-700 bg-slate-900 p-2 text-sm text-slate-100 focus:border-orange-400 focus:outline-none"
          value={context}
          onChange={(event) => setContext(event.target.value)}
          placeholder="Reason (optional)"
        />
        <select
          className="rounded-md border border-slate-700 bg-slate-900 p-2 text-sm text-slate-100 focus:border-orange-400 focus:outline-none"
          value={advantage}
          onChange={(event) => setAdvantage(event.target.value as typeof advantage)}
        >
          <option value="normal">Normal</option>
          <option value="advantage">Advantage</option>
          <option value="disadvantage">Disadvantage</option>
        </select>
        <button
          type="submit"
          className="rounded-md bg-orange-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-orange-400"
        >
          Roll
        </button>
      </form>
      <div className="mt-3 flex-1 overflow-y-auto text-sm">
        <h3 className="text-xs uppercase tracking-wide text-slate-400">Recent Rolls</h3>
        <ul className="mt-2 space-y-1">
          {diceHistory.length === 0 && <li className="text-xs text-slate-500">No rolls yet.</li>}
          {diceHistory
            .slice()
            .reverse()
            .map((result) => (
              <li key={result.timestamp} className="rounded bg-slate-900/60 px-2 py-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-orange-400">{result.total}</span>
                  <span className="text-[10px] uppercase tracking-wide text-slate-500">
                    {result.context ?? 'Roll'}
                  </span>
                </div>
                <p className="text-xs text-slate-300">{result.detail}</p>
              </li>
            ))}
        </ul>
      </div>
    </section>
  );
}
