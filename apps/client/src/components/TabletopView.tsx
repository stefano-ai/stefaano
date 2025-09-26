import { AnnouncementPayload, DiceRollRequest } from '@dnd-tabletop-sim/shared';
import { useRoomStore } from '../store/roomStore';
import {
  emitAnnouncement,
  emitDiceRoll,
  emitInitiativeUpdate,
  emitLockUpdate,
  TableSocket,
} from '../hooks/useTableSocket';
import { TabletopCanvas } from './TabletopCanvas';
import { DiceTray } from './DiceTray';
import { AnnouncementFeed } from './AnnouncementFeed';
import { GMToolbar } from './GMToolbar';

interface TabletopViewProps {
  socket: TableSocket | null;
}

export function TabletopView({ socket }: TabletopViewProps) {
  const joinDetails = useRoomStore((state) => state.joinDetails);
  const roomState = useRoomStore((state) => state.roomState);
  const diceHistory = useRoomStore((state) => state.diceHistory);
  const announcements = useRoomStore((state) => state.announcements);

  const isGM = joinDetails?.role === 'gm';

  const handleDiceRoll = (request: DiceRollRequest) => {
    emitDiceRoll(socket, request);
  };

  const handleAnnouncement = (payload: AnnouncementPayload) => {
    emitAnnouncement(socket, payload);
  };

  const handleLockToggle = (locked: boolean) => {
    emitLockUpdate(socket, { locked });
  };

  const handleInitiativeUpdate = (order: string[]) => {
    emitInitiativeUpdate(socket, { order });
  };

  if (!joinDetails || !roomState) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-900 text-slate-200">
        <p>Connecting to table...</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-slate-900 text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-800 bg-slate-950/70 px-6 py-3">
        <div>
          <h2 className="text-xl font-semibold">Room: {roomState.roomId}</h2>
          <p className="text-xs text-slate-400">Role: {joinDetails.role.toUpperCase()}</p>
        </div>
        {isGM && (
          <GMToolbar
            socket={socket}
            roomState={roomState}
            onLockToggle={handleLockToggle}
            onInitiativeUpdate={handleInitiativeUpdate}
          />
        )}
      </header>
      <main className="flex flex-1 overflow-hidden">
        <div className="flex flex-1 flex-col">
          <TabletopCanvas socket={socket} isGM={isGM} locked={roomState.locked} />
          <DiceTray
            onRoll={handleDiceRoll}
            diceHistory={diceHistory}
            rollerId={joinDetails.playerId ?? joinDetails.role}
          />
        </div>
        <aside className="flex w-80 flex-col border-l border-slate-800 bg-slate-950/40">
          <AnnouncementFeed announcements={announcements} onAnnounce={handleAnnouncement} isGM={isGM} />
        </aside>
      </main>
    </div>
  );
}

export default TabletopView;
