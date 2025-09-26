import { Token, TokenKind } from '@dnd-tabletop-sim/shared';

function randomId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2, 10);
}

function randomColor() {
  const colors = ['#f97316', '#38bdf8', '#a855f7', '#f472b6', '#22c55e'];
  return colors[Math.floor(Math.random() * colors.length)];
}

export function createToken(label: string, kind: TokenKind, ownerId?: string): Token {
  return {
    id: randomId(),
    label,
    kind,
    ownerId,
    position: { x: 100, y: 100 },
    color: randomColor(),
  };
}
