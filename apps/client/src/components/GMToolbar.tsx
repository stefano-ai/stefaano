import { FormEvent, useState } from 'react';
import { RoomState } from '@dnd-tabletop-sim/shared';
import { emitTokenAdd, TableSocket } from '../hooks/useTableSocket';
import { createToken } from '../utils/tokenFactory';

interface GMToolbarProps {
  socket: TableSocket | null;
  roomState?: RoomState;
  onLockToggle: (locked: boolean) => void;
  onInitiativeUpdate: (order: string[]) => void;
}

export function GMToolbar({ socket, roomState, onLockToggle, onInitiativeUpdate }: GMToolbarProps) {
  const [npcName, setNpcName] = useState('Goblin');
  const [initiativeList, setInitiativeList] = useState('');

  const handleAddNpc = () => {
    const label = npcName.trim() || 'NPC';
    const token = createToken(label, 'npc');
    emitTokenAdd(socket, { token });
    setNpcName('');
  };

  const handleLockChange = () => {
    if (!roomState) return;
    onLockToggle(!roomState.locked);
  };

  const handleInitiativeSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const order = initiativeList
      .split('\n')
      .map((entry) => entry.trim())
      .filter(Boolean);
    if (order.length > 0) {
      onInitiativeUpdate(order);
      setInitiativeList('');
    }
  };

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <input
          className="w-32 rounded-md border border-slate-700 bg-slate-900 p-1 text-sm text-slate-100 focus:border-orange-400 focus:outline-none"
          value={npcName}
          onChange={(event) => setNpcName(event.target.value)}
          placeholder="NPC name"
        />
        <button
          type="button"
          className="rounded-md bg-slate-800 px-3 py-1 text-sm font-semibold text-slate-100 hover:bg-slate-700"
          onClick={handleAddNpc}
        >
          Add NPC
        </button>
      </div>
      <button
        type="button"
        className="rounded-md border border-orange-400 px-3 py-1 text-sm font-semibold text-orange-400 hover:bg-orange-500 hover:text-white"
        onClick={handleLockChange}
      >
        {roomState?.locked ? 'Unlock Room' : 'Lock Room'}
      </button>
      <form className="flex items-center gap-2" onSubmit={handleInitiativeSubmit}>
        <textarea
          className="h-16 w-40 rounded-md border border-slate-700 bg-slate-900 p-2 text-xs text-slate-100 focus:border-orange-400 focus:outline-none"
          placeholder="Initiative order (one per line)"
          value={initiativeList}
          onChange={(event) => setInitiativeList(event.target.value)}
        />
        <button
          type="submit"
          className="rounded-md bg-orange-500 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white hover:bg-orange-400"
        >
          Update
        </button>
      </form>
    </div>
  );
}
