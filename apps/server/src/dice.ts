import { DiceRollRequest, DiceRollResult } from '@dnd-tabletop-sim/shared';

interface ParsedFormula {
  count: number;
  sides: number;
  modifier: number;
}

const FORMULA_REGEX = /^(\d*)d(\d+)([+-]\d+)?$/i;

function parseFormula(formula: string): ParsedFormula {
  const trimmed = formula.replace(/\s+/g, '');
  const match = FORMULA_REGEX.exec(trimmed);

  if (!match) {
    throw new Error(`Invalid dice formula: ${formula}`);
  }

  const [, countRaw, sidesRaw, modifierRaw] = match;
  const count = countRaw ? Number.parseInt(countRaw, 10) : 1;
  const sides = Number.parseInt(sidesRaw, 10);
  const modifier = modifierRaw ? Number.parseInt(modifierRaw, 10) : 0;

  if (Number.isNaN(count) || Number.isNaN(sides) || Number.isNaN(modifier)) {
    throw new Error(`Invalid dice formula: ${formula}`);
  }

  return { count, sides, modifier };
}

function rollSingleDie(sides: number): number {
  return Math.floor(Math.random() * sides) + 1;
}

function rollWithAdvantage(sides: number, takeHighest: boolean): { rolls: number[]; value: number } {
  const first = rollSingleDie(sides);
  const second = rollSingleDie(sides);
  const rolls = [first, second];
  const value = takeHighest ? Math.max(first, second) : Math.min(first, second);
  return { rolls, value };
}

export function rollDice(request: DiceRollRequest): DiceRollResult {
  const parsed = parseFormula(request.formula);
  const { count, sides, modifier } = parsed;
  const rolls: number[] = [];
  const detailedParts: string[] = [];
  let total = modifier;

  for (let index = 0; index < count; index += 1) {
    if (sides === 20 && count === 1 && (request.advantage || request.disadvantage)) {
      const { rolls: advRolls, value } = rollWithAdvantage(sides, Boolean(request.advantage));
      rolls.push(...advRolls);
      detailedParts.push(`${request.advantage ? 'adv' : 'dis'}(${advRolls.join(',')})`);
      total += value;
    } else {
      const roll = rollSingleDie(sides);
      rolls.push(roll);
      detailedParts.push(`${roll}`);
      total += roll;
    }
  }

  if (modifier !== 0) {
    detailedParts.push(modifier > 0 ? `+${modifier}` : `${modifier}`);
  }

  return {
    ...request,
    formula: request.formula,
    rolls,
    total,
    detail: `${count}d${sides}${modifier >= 0 ? `+${modifier}` : modifier} => ${detailedParts.join(' + ')}`,
    timestamp: new Date().toISOString(),
  };
}
